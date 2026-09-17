"""Persistent Recurrent Memory module.

Memory M has shape (batch, n_slots, d_mem) and is initialized to 0.

Per token t (scan over the sequence):
  1. Observe: M' = CrossAttn(Q=M, K=H_{<=t}, V=H_{<=t}), then project to d_mem.
  2. Update:  M <- GRU(M', M) per slot with shared weights.
  3. Influence: pool slots by mean, then
       H_t <- H_t + σ(W_g m) ⊙ (W_p m)
     with W_g, W_p : d_mem -> d_model.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax_prm.attn import cross_attend_memory, dropout, init_cross_attn
from jax_prm.config import PRMConfig
from jax_prm.gru import gru, init_gru

Array = jax.Array
Params = dict[str, Any]


def init_prm(key: Array, cfg: PRMConfig) -> Params:
    k_attn, k_gru, k_g, k_p, k_aux = jax.random.split(key, 5)
    params: Params = init_cross_attn(k_attn, cfg.d_mem, cfg.d_model)
    params["gru"] = init_gru(k_gru, cfg.d_mem)
    std = cfg.d_mem**-0.5
    params["Wg"] = jax.random.normal(k_g, (cfg.d_mem, cfg.d_model)) * std
    params["Wp"] = jax.random.normal(k_p, (cfg.d_mem, cfg.d_model)) * std
    if cfg.aux_lambda > 0:
        params["W_aux"] = jax.random.normal(k_aux, (cfg.d_mem, cfg.d_model)) * std
    return params


def init_memory(batch: int, cfg: PRMConfig, dtype=jnp.float32) -> Array:
    return jnp.zeros((batch, cfg.n_slots, cfg.d_mem), dtype=dtype)


def gated_add(wg: Array, wp: Array, h: Array, m: Array) -> Array:
    """H_out = H + σ(W_g m) ⊙ (W_p m). Last axis of m is d_mem, of h is d_model."""
    return h + jax.nn.sigmoid(m @ wg) * (m @ wp)


def prm_forward(
    params: Params,
    h: Array,
    cfg: PRMConfig,
    *,
    memory: Array | None = None,
    key: Array | None = None,
    training: bool = False,
) -> tuple[Array, Array]:
    """Run PRM over a hidden sequence.

    h: (B, T, d_model). Returns (H_out, M_final) with M_final (B, n_slots, d_mem).
    Memory is recurrent across tokens: at step t the GRU state is the state after t-1,
    and cross-attention is causally masked to H_{<=t}.
    """
    b, t_len, _ = h.shape
    if memory is None:
        memory = init_memory(b, cfg, dtype=h.dtype)

    use_drop = bool(training and cfg.dropout > 0.0 and key is not None)
    rng0 = key if key is not None else jax.random.PRNGKey(0)
    n_heads = cfg.n_mem_heads
    wg = params["Wg"]
    wp = params["Wp"]
    gru_params = params["gru"]
    drop = cfg.dropout

    def _step(carry, t_index):
        m, rng = carry
        m_obs = cross_attend_memory(params, m, h, n_heads, t_index=t_index)
        if use_drop:
            rng, k1, k2 = jax.random.split(rng, 3)
            m_obs = dropout(m_obs, drop, k1)
        else:
            k2 = None
        m_new = gru(gru_params, m_obs, m)
        pooled = jnp.mean(m_new, axis=1)
        delta = jax.nn.sigmoid(pooled @ wg) * (pooled @ wp)
        if use_drop:
            delta = dropout(delta, drop, k2)
        return (m_new, rng), delta

    (m_final, _), deltas = jax.lax.scan(_step, (memory, rng0), jnp.arange(t_len))
    # deltas: (T, B, d_model)
    deltas = jnp.swapaxes(deltas, 0, 1)
    return h + deltas, m_final
