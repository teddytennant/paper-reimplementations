"""Identity checks for HeRo routing, memory, gates, and skip loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hero.config import HeroConfig
from hero.model import forward, init_params, total_loss
from hero.ops import memory_write, phi, routed_residual, skip_loss
from hero.router import init_shared_router, route_block


def _cfg() -> HeroConfig:
    return HeroConfig(n_layers=4, d_model=32, vocab_size=32)


def _run():
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 6), 0, cfg.vocab_size)
    _, aux = forward(params, tokens, cfg, return_aux=True)
    return cfg, params, tokens, aux


def test_read_before_write_zero_context_at_first_block():
    _, _, _, aux = _run()
    c0 = aux["c"][..., 0]
    np.testing.assert_allclose(c0, 0.0, atol=1e-6)


def test_memory_recurrence_matches_outer_product_update():
    cfg = _cfg()
    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    d_m = cfg.d_m
    s_prev = 0.2 * jax.random.normal(k1, (3, d_m, d_m), dtype=jnp.float32)
    zeta_prev = 0.2 * jax.random.normal(k2, (3, d_m), dtype=jnp.float32)
    k_vec = jax.random.normal(k3, (3, d_m), dtype=jnp.float32)
    v = jax.random.normal(k4, (3, d_m), dtype=jnp.float32)
    rho = jnp.array(0.7, dtype=jnp.float32)
    phi_k = phi(k_vec)
    s_new, zeta_new = memory_write(s_prev, zeta_prev, phi_k, v, rho)
    s_ref = rho * s_prev + phi_k[:, :, None] * v[:, None, :]
    zeta_ref = rho * zeta_prev + phi_k
    np.testing.assert_allclose(s_new, s_ref, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(zeta_new, zeta_ref, atol=1e-6, rtol=1e-5)


def test_first_layer_nu_is_identically_zero():
    _, _, _, aux = _run()
    np.testing.assert_allclose(aux["nu"][..., 0], 0.0, atol=0.0)


def test_cumulative_history_is_running_product_from_one():
    _, _, _, aux = _run()
    p = aux["p_hist"]
    m = aux["m_hist"]
    np.testing.assert_allclose(m[..., 0], p[..., 0], atol=1e-6, rtol=1e-5)
    running = jnp.cumprod(p, axis=-1)
    np.testing.assert_allclose(m, running, atol=1e-6, rtol=1e-5)


def test_gate_is_sigmoid_of_summed_logits():
    _, _, _, aux = _run()
    expected = jax.nn.sigmoid(aux["a"] + aux["b"])
    np.testing.assert_allclose(aux["gates"], expected, atol=1e-6, rtol=1e-5)


def test_branch_selection_threshold():
    hbar = jnp.zeros((2, 4), dtype=jnp.float32)
    ffn_out = jnp.ones((2, 4), dtype=jnp.float32)
    adapter_out = 3.0 * jnp.ones((2, 4), dtype=jnp.float32)
    tau = 0.5
    g_hi = jnp.array([0.8, 0.9], dtype=jnp.float32)
    g_lo = jnp.array([0.1, 0.4], dtype=jnp.float32)
    out_hi = routed_residual(hbar, g_hi, ffn_out, adapter_out, tau)
    out_lo = routed_residual(hbar, g_lo, ffn_out, adapter_out, tau)
    np.testing.assert_allclose(out_hi, g_hi[:, None] * ffn_out, atol=1e-6)
    np.testing.assert_allclose(out_lo, (1.0 - g_lo)[:, None] * adapter_out, atol=1e-6)


def test_skip_loss_matches_mean_squared_token_gate_sum():
    gates = jnp.array(
        [
            [[0.2, 0.3, 0.1], [0.5, 0.0, 0.4]],
            [[0.9, 0.1, 0.2], [0.3, 0.3, 0.3]],
        ],
        dtype=jnp.float32,
    )
    got = skip_loss(gates)
    token_sums = jnp.sum(gates, axis=-1)
    ref = jnp.mean(token_sums**2)
    np.testing.assert_allclose(got, ref, atol=1e-6)
    b, t_pred, _ = gates.shape
    ref_explicit = jnp.sum(token_sums**2) / (b * t_pred)
    np.testing.assert_allclose(got, ref_explicit, atol=1e-6)


def test_autodiff_skip_loss_matches_finite_difference():
    cfg = _cfg()
    params = init_params(jax.random.PRNGKey(0), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, 5), 0, cfg.vocab_size)

    def skip_from_bias(bias: jax.Array) -> jax.Array:
        def replace_bias(p):
            router = dict(p["router"])
            router["b_theta"] = jnp.reshape(bias, (1,))
            return {**p, "router": router}

        _, aux = forward(replace_bias(params), tokens, cfg, return_aux=True)
        return skip_loss(aux["gates"][:, :-1, :])

    bias0 = params["router"]["b_theta"][0]
    grad = jax.grad(skip_from_bias)(bias0)
    eps = jnp.float32(1e-3)
    fd = (skip_from_bias(bias0 + eps) - skip_from_bias(bias0 - eps)) / (2.0 * eps)
    np.testing.assert_allclose(grad, fd, rtol=2e-2, atol=2e-3)


def test_skip_loss_couples_layers():
    gates = jnp.array(
        [[[0.2, 0.3, 0.4], [0.1, 0.5, 0.2]]],
        dtype=jnp.float32,
    )
    base = skip_loss(gates)
    bumped = skip_loss(gates.at[..., 0].add(0.25))
    assert float(bumped) > float(base)


def test_phi_is_elu_plus_one_and_strictly_positive():
    moderate = jnp.linspace(-4.0, 8.0, 241, dtype=jnp.float32)
    np.testing.assert_allclose(phi(moderate), jax.nn.elu(moderate) + 1.0, atol=1e-6, rtol=1e-5)
    x = jnp.linspace(-20.0, 20.0, 401, dtype=jnp.float32)
    got = phi(x)
    assert bool(jnp.all(got > 0.0))


def test_route_block_zero_memory_gives_zero_context():
    cfg = _cfg()
    router = init_shared_router(jax.random.PRNGKey(4), cfg)
    hbar = jax.random.normal(jax.random.PRNGKey(5), (2, 3, cfg.d_model), dtype=jnp.float32)
    zeros_s = jnp.zeros((2, 3, cfg.d_m, cfg.d_m), dtype=jnp.float32)
    zeros_z = jnp.zeros((2, 3, cfg.d_m), dtype=jnp.float32)
    ones = jnp.ones((2, 3), dtype=jnp.float32)
    step = route_block(
        hbar,
        1,
        cfg.n_routed,
        router,
        jnp.array(0.0, dtype=jnp.float32),
        zeros_s,
        zeros_z,
        None,
        None,
        ones,
        ones,
        cfg,
    )
    np.testing.assert_allclose(step["c"], 0.0, atol=1e-6)
    np.testing.assert_allclose(step["nu"], 0.0, atol=0.0)
    s_ref, z_ref = memory_write(
        zeros_s, zeros_z, step["phi_k"], step["v"], step["rho"]
    )
    np.testing.assert_allclose(step["S"], s_ref, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(step["zeta"], z_ref, atol=1e-6, rtol=1e-5)


def test_total_loss_is_finite():
    cfg, params, tokens, _ = _run()
    loss, aux = total_loss(params, tokens, cfg)
    assert jnp.isfinite(loss)
    np.testing.assert_allclose(aux["skip"], skip_loss(aux["gates"][:, :-1, :]), atol=1e-6)
