"""GRU memory update (paper §2, four equations).

Applied per slot with shared weights. Concatenate is on the last axis.

    z     = σ(W_z [M', M] + b_z)
    r     = σ(W_r [M', M] + b_r)
    M̃    = tanh(W_h [M', r ⊙ M] + b_h)
    M_new = (1 - z) ⊙ M + z ⊙ M̃
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

Array = jax.Array
Params = dict[str, Any]


def init_gru(key: Array, d_mem: int) -> Params:
    kz, kr, kh = jax.random.split(key, 3)
    std = (2 * d_mem) ** -0.5
    return {
        "Wz": jax.random.normal(kz, (2 * d_mem, d_mem)) * std,
        "Wr": jax.random.normal(kr, (2 * d_mem, d_mem)) * std,
        "Wh": jax.random.normal(kh, (2 * d_mem, d_mem)) * std,
        "bz": jnp.zeros((d_mem,), dtype=jnp.float32),
        "br": jnp.zeros((d_mem,), dtype=jnp.float32),
        "bh": jnp.zeros((d_mem,), dtype=jnp.float32),
    }


def gru(params: Params, m_obs: Array, m: Array) -> Array:
    """One GRU step. m_obs is M', m is M. Leading dims (e.g. batch, slots) broadcast."""
    concat = jnp.concatenate([m_obs, m], axis=-1)
    z = jax.nn.sigmoid(concat @ params["Wz"] + params["bz"])
    r = jax.nn.sigmoid(concat @ params["Wr"] + params["br"])
    candidate_in = jnp.concatenate([m_obs, r * m], axis=-1)
    m_tilde = jnp.tanh(candidate_in @ params["Wh"] + params["bh"])
    return (1.0 - z) * m + z * m_tilde
