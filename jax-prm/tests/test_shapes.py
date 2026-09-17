"""Shape checks for GRU, PRM, attention, and the full decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_prm.attn import cross_attend_memory, self_attention
from jax_prm.config import PRMConfig
from jax_prm.gru import gru, init_gru
from jax_prm.prm import gated_add, init_memory, init_prm, prm_forward
from jax_prm.transformer import forward, init


def test_gru_shape():
    d = 8
    params = init_gru(jax.random.PRNGKey(0), d)
    m = jax.random.normal(jax.random.PRNGKey(1), (3, 5, d))
    m_obs = jax.random.normal(jax.random.PRNGKey(2), (3, 5, d))
    out = gru(params, m_obs, m)
    assert out.shape == (3, 5, d)


def test_prm_and_gated_add_shapes():
    cfg = PRMConfig(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=16,
        context=16,
        n_slots=4,
        d_mem=16,
        n_mem_heads=4,
        dropout=0.0,
    )
    params = init_prm(jax.random.PRNGKey(0), cfg)
    h = jax.random.normal(jax.random.PRNGKey(1), (2, 7, cfg.d_model))
    h_out, mem = prm_forward(params, h, cfg)
    assert h_out.shape == h.shape
    assert mem.shape == (2, cfg.n_slots, cfg.d_mem)
    m0 = init_memory(2, cfg)
    assert m0.shape == (2, 4, 16)
    pooled = mem.mean(axis=1)
    ga = gated_add(params["Wg"], params["Wp"], h[:, 0], pooled)
    assert ga.shape == (2, cfg.d_model)


def test_attention_shapes():
    cfg = PRMConfig(d_model=32, n_heads=4, n_mem_heads=4, d_mem=16, n_slots=3, dropout=0.0)
    params = init(jax.random.PRNGKey(0), cfg)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 32))
    out, w = self_attention(params["layers"][0]["attn"], x, 4)
    assert out.shape == (2, 5, 32)
    assert w.shape == (2, 4, 5, 5)
    mem = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
    m_obs = cross_attend_memory(params["prm"], mem, x, 4, t_index=jnp.array(2))
    assert m_obs.shape == (2, 3, 16)


def test_transformer_logits_shape():
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
    params = init(jax.random.PRNGKey(0), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, 32)
    logits, info = forward(params, tokens, cfg)
    assert logits.shape == (2, 8, 32)
    assert info["h"].shape == (2, 8, 32)
    assert info["h_lower"].shape == (2, 8, 32)
    assert info["h_prm"].shape == (2, 8, 32)
    assert info["memory"].shape == (2, 4, 16)
    assert info["attn_weights"].shape[0] == 2
