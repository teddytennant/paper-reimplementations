"""Shapes, causal mask, CoT skip-embed, and attention probabilities (eqs 1–9)."""

import jax
import jax.numpy as jnp

from jax_atc.model import ATCConfig, assemble_embeddings, forward, init_params


def _theoretical(n=6, d=16, max_phases=6):
    return ATCConfig(d=d, n=n, max_phases=max_phases)


def test_shapes_phase3():
    n, d, phase = 6, 16, 3
    config = _theoretical(n=n, d=d)
    key = jax.random.PRNGKey(0)
    params = init_params(key, config)
    bits = jnp.ones((n,))
    bits = bits.at[:phase].set(jnp.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[:phase])
    cots = jax.random.normal(jax.random.PRNGKey(1), (phase - 1, d))
    res = forward(params, bits, cots, phase, config)
    # n + (r-1) CoT + 1 final = 9
    assert res.embeddings.shape == (9, d)
    assert res.outputs.shape == (9, d)
    assert res.attn.shape == (9, 9)
    assert res.yhat.shape == ()


def test_causal_no_future_mass():
    config = _theoretical(n=6, d=16)
    params = init_params(jax.random.PRNGKey(2), config)
    bits = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    cots = jax.random.normal(jax.random.PRNGKey(3), (2, config.d))
    res = forward(params, bits, cots, phase=3, config=config)
    future = jnp.triu(res.attn, k=1)
    assert jnp.max(jnp.abs(future)) < 1e-6


def test_cot_skip_embed():
    config = _theoretical(n=4, d=8, max_phases=4)
    params = init_params(jax.random.PRNGKey(4), config)
    bits = jnp.array([1.0, -1.0, 1.0, 1.0])
    z = jax.random.normal(jax.random.PRNGKey(5), (1, config.d))
    emb = assemble_embeddings(params, bits, z, config)
    assert emb.shape == (4 + 1 + 1, config.d)
    assert jnp.allclose(emb[4], z[0], atol=1e-6)

    params_E = dict(params)
    params_E["E"] = params["E"] + 10.0
    emb2 = assemble_embeddings(params_E, bits, z, config)
    assert jnp.allclose(emb2[4], z[0], atol=1e-6)
    assert not jnp.allclose(emb2[0], emb[0], atol=1e-3)


def test_final_attn_is_probability():
    config = _theoretical(n=6, d=16)
    params = init_params(jax.random.PRNGKey(6), config)
    bits = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    cots = jax.random.normal(jax.random.PRNGKey(7), (2, config.d))
    res = forward(params, bits, cots, phase=3, config=config)
    alpha_final = res.attn[-1]
    assert jnp.all(alpha_final >= -1e-6)
    assert jnp.allclose(alpha_final.sum(), 1.0, atol=1e-5)
