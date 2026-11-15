"""Matrix-valued trust operators P_I and P_M (RiVaT-Fuse eqs 8-9).

P = Diag(d(s)) + U Diag(g(s)) U^T + ε I

d(s) and g(s) are affine + softplus so the diagonal weights stay positive.
U is a learned d x r_rank factor. ε > 0 keeps the operator PD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS_DEFAULT = 1e-4
POS_FLOOR = 1e-6


def _normal(key, shape, scale=0.1):
    return scale * jax.random.normal(key, shape, dtype=jnp.float32)


def softplus_pos(x, floor=POS_FLOOR):
    return jax.nn.softplus(x) + floor


def init_trust_side(key, d, r_rank, s_dim):
    """Learned U and affine maps for one modality's trust operator."""
    kU, kd, kg = jax.random.split(key, 3)
    return {
        "U": _normal(kU, (d, r_rank), scale=0.1),
        "d_W": _normal(kd, (d, s_dim), scale=0.05),
        "d_b": jnp.zeros((d,), dtype=jnp.float32),
        "g_W": _normal(kg, (r_rank, s_dim), scale=0.05),
        "g_b": jnp.zeros((r_rank,), dtype=jnp.float32),
    }


def init_trust_params(key, d, r_rank, s_dim):
    kI, kM = jax.random.split(key)
    return {
        "I": init_trust_side(kI, d, r_rank, s_dim),
        "M": init_trust_side(kM, d, r_rank, s_dim),
    }


def maps_d_g(side, s):
    """d(s) in R^d and g(s) in R^{r_rank}, both strictly positive."""
    d_vec = softplus_pos(side["d_W"] @ s + side["d_b"])
    g_vec = softplus_pos(side["g_W"] @ s + side["g_b"])
    return d_vec, g_vec


def dense_P(d_vec, g_vec, U, eps=EPS_DEFAULT):
    """Dense d x d matrix Diag(d) + U Diag(g) U^T + ε I."""
    dim = d_vec.shape[0]
    low_rank = (U * g_vec[None, :]) @ U.T
    return jnp.diag(d_vec) + low_rank + eps * jnp.eye(dim, dtype=d_vec.dtype)


def apply_P(d_vec, g_vec, U, v, eps=EPS_DEFAULT):
    """Matvec (Diag(d) + U Diag(g) U^T + ε I) v without forming P."""
    return d_vec * v + U @ (g_vec * (U.T @ v)) + eps * v


def dense_P_from_side(side, s, eps=EPS_DEFAULT):
    d_vec, g_vec = maps_d_g(side, s)
    return dense_P(d_vec, g_vec, side["U"], eps=eps)


def apply_P_from_side(side, s, v, eps=EPS_DEFAULT):
    d_vec, g_vec = maps_d_g(side, s)
    return apply_P(d_vec, g_vec, side["U"], v, eps=eps)
