"""Numerical checks of the PRM equations (arXiv:2609.17251) on tiny tensors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_prm.attn import self_attention
from jax_prm.config import PRMConfig
from jax_prm.gru import gru
from jax_prm.prm import gated_add, prm_forward
from jax_prm.transformer import count_params, init


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_gru_four_equations_hand_computed():
    """z, r, candidate, interpolant match a numpy evaluation of the four equations."""
    rng = np.random.default_rng(0)
    d = 3
    m = rng.normal(size=(2, d)).astype(np.float32)
    m_obs = rng.normal(size=(2, d)).astype(np.float32)
    wz = rng.normal(size=(2 * d, d)).astype(np.float32) * 0.4
    wr = rng.normal(size=(2 * d, d)).astype(np.float32) * 0.4
    wh = rng.normal(size=(2 * d, d)).astype(np.float32) * 0.4
    bz = rng.normal(size=(d,)).astype(np.float32) * 0.1
    br = rng.normal(size=(d,)).astype(np.float32) * 0.1
    bh = rng.normal(size=(d,)).astype(np.float32) * 0.1

    concat = np.concatenate([m_obs, m], axis=-1)
    z = _sigmoid(concat @ wz + bz)
    r = _sigmoid(concat @ wr + br)
    m_tilde = np.tanh(np.concatenate([m_obs, r * m], axis=-1) @ wh + bh)
    expected = (1.0 - z) * m + z * m_tilde

    params = {
        "Wz": jnp.asarray(wz),
        "Wr": jnp.asarray(wr),
        "Wh": jnp.asarray(wh),
        "bz": jnp.asarray(bz),
        "br": jnp.asarray(br),
        "bh": jnp.asarray(bh),
    }
    got = np.asarray(gru(params, jnp.asarray(m_obs), jnp.asarray(m)))
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_gated_addition_toy_linear():
    """H_out = H + sigmoid(Wg m) * (Wp m) with no other ops."""
    rng = np.random.default_rng(1)
    b, t, d_model, d_mem = 2, 4, 6, 5
    h = rng.normal(size=(b, t, d_model)).astype(np.float32)
    m = rng.normal(size=(b, t, d_mem)).astype(np.float32)
    wg = rng.normal(size=(d_mem, d_model)).astype(np.float32) * 0.3
    wp = rng.normal(size=(d_mem, d_model)).astype(np.float32) * 0.3
    expected = h + _sigmoid(m @ wg) * (m @ wp)
    got = np.asarray(gated_add(jnp.asarray(wg), jnp.asarray(wp), jnp.asarray(h), jnp.asarray(m)))
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_prm_causality_differ_at_last_position():
    """Sequences that differ only at t=T-1 match at positions < T-1 after PRM."""
    cfg = PRMConfig(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=32,
        context=16,
        n_slots=4,
        d_mem=16,
        n_mem_heads=4,
        dropout=0.0,
    )
    key = jax.random.PRNGKey(2)
    params = init(key, cfg)["prm"]
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    h1 = jax.random.normal(k1, (2, 8, cfg.d_model))
    h2 = h1.at[:, -1, :].set(jax.random.normal(k2, (2, cfg.d_model)))
    out1, _ = prm_forward(params, h1, cfg)
    out2, _ = prm_forward(params, h2, cfg)
    np.testing.assert_allclose(np.asarray(out1[:, :-1]), np.asarray(out2[:, :-1]), atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(out1[:, -1] - out2[:, -1]))) > 1e-5


def test_paper_scale_param_overhead():
    """PRM extra params are a few percent of the baseline transformer at paper dims."""
    cfg_base = PRMConfig(
        n_layers=6,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        vocab_size=8000,
        context=256,
        n_slots=16,
        d_mem=128,
        n_mem_heads=4,
        dropout=0.0,
        use_prm=False,
    )
    cfg_prm = PRMConfig(
        n_layers=6,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        vocab_size=8000,
        context=256,
        n_slots=16,
        d_mem=128,
        n_mem_heads=4,
        dropout=0.0,
        use_prm=True,
    )
    key = jax.random.PRNGKey(0)
    base = count_params(init(key, cfg_base))
    full = count_params(init(key, cfg_prm))
    extra = full - base
    ratio = extra / base
    assert extra > 0
    assert 0.02 < ratio < 0.06, f"overhead {ratio:.4f} not in (0.02, 0.06); extra={extra} base={base}"


def test_causal_self_attn_no_future():
    """Position 0 cannot attend to position 1 (upper triangle of weights is ~0)."""
    cfg = PRMConfig(d_model=32, n_heads=4, dropout=0.0)
    params = init(jax.random.PRNGKey(4), cfg)
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 6, cfg.d_model))
    _out, weights = self_attention(params["layers"][0]["attn"], x, cfg.n_heads)
    # weights: (B, H, T, T)
    t = weights.shape[-1]
    future = jnp.triu(jnp.ones((t, t), dtype=weights.dtype), k=1)
    leaked = jnp.max(jnp.abs(weights * future[None, None, :, :]))
    assert float(leaked) < 1e-6
    # differ-at-end: prefix of the attended output is independent of the last token
    x2 = x.at[:, -1, :].set(jax.random.normal(jax.random.PRNGKey(6), (2, cfg.d_model)))
    out1, _ = self_attention(params["layers"][0]["attn"], x, cfg.n_heads)
    out2, _ = self_attention(params["layers"][0]["attn"], x2, cfg.n_heads)
    np.testing.assert_allclose(np.asarray(out1[:, :-1]), np.asarray(out2[:, :-1]), atol=1e-5, rtol=1e-5)
