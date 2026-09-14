import jax
import jax.numpy as jnp

from klxx.flow import forward, init_coupling, inverse, source_potential, two_mode_potential
from klxx.loss import klxx_loss, log_ratio_variation, z_from_flow
from klxx.sampling import mix_xi, qt, sample_two_mode, smc


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(grads, params, state, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}


def test_mix_xi_alpha1_uses_only_qt_and_alpha0_uses_only_pushforward():
    key = jax.random.PRNGKey(3)
    key, k_qt, k_nu, k_mix = jax.random.split(key, 4)
    qt_y = jax.random.normal(k_qt, (16, 2)) + jnp.array([2.0, 0.0])
    nu_y = jax.random.normal(k_nu, (16, 2))
    only_qt = mix_xi(qt_y, nu_y, k_mix, alpha=1.0, beta=0.0)
    only_nu = mix_xi(qt_y, nu_y, k_mix, alpha=0.0, beta=1.0)
    assert jnp.allclose(only_qt, qt_y)
    assert jnp.allclose(only_nu, nu_y)

    params = init_coupling(key, dim=2, n_layers=2, hidden=8)
    z_qt = z_from_flow(params, only_qt, two_mode_potential, source_potential, forward)
    z_nu = z_from_flow(params, only_nu, two_mode_potential, source_potential, forward)
    z_xi_qt = z_from_flow(
        params, mix_xi(qt_y, nu_y, k_mix, alpha=1.0, beta=0.0), two_mode_potential, source_potential, forward
    )
    z_xi_nu = z_from_flow(
        params, mix_xi(qt_y, nu_y, k_mix, alpha=0.0, beta=1.0), two_mode_potential, source_potential, forward
    )
    assert jnp.allclose(log_ratio_variation(z_xi_qt), log_ratio_variation(z_qt), atol=1e-6)
    assert jnp.allclose(log_ratio_variation(z_xi_nu), log_ratio_variation(z_nu), atol=1e-6)


def test_smc_returns_finite_same_shape():
    key = jax.random.PRNGKey(4)
    key, k_init, k_x, k_smc = jax.random.split(key, 4)
    params = init_coupling(k_init, dim=2, n_layers=2, hidden=8)
    x = jax.random.normal(k_x, (32, 2))
    y, _ = inverse(params, x)
    z_fn = lambda yy: z_from_flow(params, yy, two_mode_potential, source_potential, forward)
    y_hat = smc(y, z_fn, two_mode_potential, k_smc, M=3, mala_steps=2)
    assert y_hat.shape == y.shape
    assert bool(jnp.all(jnp.isfinite(y_hat)))


def test_klxx_train_finite_and_progress():
    key = jax.random.PRNGKey(5)
    key, k_init, k_pi, k_nu, k_qt, k_mix = jax.random.split(key, 6)
    params = init_coupling(k_init, dim=2, n_layers=4, hidden=32)
    y_pi = sample_two_mode(k_pi, 64)
    x0 = jax.random.normal(k_nu, (64, 2))
    y_nu, _ = inverse(params, x0)
    y_qt = qt(x0, two_mode_potential, k_qt, melt=2.5, quench_steps=20, mala_steps=5)
    y_xi = mix_xi(y_qt, y_nu, k_mix, alpha=0.5, beta=0.5)

    def loss_fn(p, y_pi, y_xi):
        z_pi = z_from_flow(p, y_pi, two_mode_potential, source_potential, forward)
        z_xi = z_from_flow(p, y_xi, two_mode_potential, source_potential, forward)
        return klxx_loss(z_pi, z_xi, lam=1.0, theta=1.0)

    loss_before = loss_fn(params, y_pi, y_xi)
    state = _adam_init(params)

    @jax.jit
    def step(params, state, y_pi, y_xi):
        loss, grads = jax.value_and_grad(loss_fn)(params, y_pi, y_xi)
        params, state = _adam_update(grads, params, state, lr=0.02)
        return params, state, loss

    loss = loss_before
    for i in range(40):
        key, k_pi, k_nu, k_mix = jax.random.split(key, 4)
        y_pi = sample_two_mode(k_pi, 64)
        x0 = jax.random.normal(k_nu, (64, 2))
        y_nu, _ = inverse(params, x0)
        y_nu = jax.lax.stop_gradient(y_nu)
        y_xi = mix_xi(jax.lax.stop_gradient(y_qt), y_nu, k_mix, alpha=0.5, beta=0.5)
        params, state, loss = step(params, state, y_pi, y_xi)

    loss_after = loss
    assert bool(jnp.isfinite(loss_before))
    assert bool(jnp.isfinite(loss_after))
    key, k_gen = jax.random.split(key)
    x_gen = jax.random.normal(k_gen, (256, 2))
    y_gen, _ = inverse(params, x_gen)
    modes = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
    hits = 0
    for m in modes:
        if bool(jnp.any(jnp.linalg.norm(y_gen - m, axis=-1) < 1.0)):
            hits += 1
    assert (float(loss_after) < float(loss_before)) or (hits >= 2)
