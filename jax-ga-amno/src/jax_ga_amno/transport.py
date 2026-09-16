"""Gauge-aware transport (paper Sec. 3.4 / Eqs. 8–10).

T_{i←j} = diag(d_ij) + (γ_ij / sqrt(C_h r)) U_ij V_ij^T
α_ij    = softmax_j(a_ij) over N(i)
m_i     = sum_{j in N(i)} α_ij T_{i←j} h_j
h_i^{ℓ+1} = h_i^ℓ + ρ_θ(m_i)
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def transport_scale(hidden: int, rank: int, dtype=jnp.float32) -> Array:
    """1 / sqrt(C_h r) from Eq. 8."""
    return jnp.asarray(1.0 / jnp.sqrt(hidden * rank), dtype=dtype)


def transport_matrix(
    d: Array,
    u: Array,
    v: Array,
    gamma: Array,
) -> Array:
    """T = diag(d) + (γ / sqrt(C_h r)) U V^T, shape (..., C_h, C_h)."""
    hidden = d.shape[-1]
    rank = u.shape[-1]
    scale = transport_scale(hidden, rank, dtype=d.dtype)
    eye = jnp.eye(hidden, dtype=d.dtype)
    diag = eye * d[..., None]
    uv = jnp.einsum("...cr,...dr->...cd", u, v)
    g = jnp.reshape(gamma, gamma.shape + (1,) * (uv.ndim - gamma.ndim))
    return diag + scale * g * uv


def apply_transport(
    d: Array,
    u: Array,
    v: Array,
    gamma: Array,
    h: Array,
) -> Array:
    """(T h) without materialising T: d ⊙ h + (γ / sqrt(C_h r)) U (V^T h)."""
    hidden = h.shape[-1]
    rank = u.shape[-1]
    scale = transport_scale(hidden, rank, dtype=h.dtype)
    g = jnp.reshape(gamma, gamma.shape + (1,) * (h.ndim - gamma.ndim))
    vt_h = jnp.einsum("...cr,...c->...r", v, h)
    low = jnp.einsum("...cr,...r->...c", u, vt_h)
    return d * h + scale * g * low


def attention_aggregate(
    transported: Array,
    scores: Array,
    mask: Array | None = None,
) -> tuple[Array, Array]:
    """α = softmax_j(a_ij); m_i = sum_j α_ij (T_{i←j} h_j). Eq. 9.

    ``transported`` is (..., N, K, C), ``scores`` is (..., N, K).
    """
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.asarray(-1e9, dtype=scores.dtype))
    alpha = jax.nn.softmax(scores, axis=-1)
    if mask is not None:
        alpha = jnp.where(mask, alpha, jnp.asarray(0.0, dtype=alpha.dtype))
    m = jnp.sum(alpha[..., None] * transported, axis=-2)
    return m, alpha


def residual_update(
    h: Array,
    m: Array,
    rho: Callable[[Array], Array] | None = None,
) -> Array:
    """h^{ℓ+1} = h^ℓ + ρ_θ(m). ρ identity if ``rho`` is None. Eq. 10."""
    delta = m if rho is None else rho(m)
    return h + delta


def transport_heads(params: dict, edges: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Linear heads on e_ij → (d, U, V, γ, a). γ is a sigmoid gate."""
    d = edges @ params["W_d"] + params["b_d"]
    hidden = d.shape[-1]
    rank = params["W_U"].shape[-1] // hidden
    u = (edges @ params["W_U"] + params["b_U"]).reshape(edges.shape[:-1] + (hidden, rank))
    v = (edges @ params["W_V"] + params["b_V"]).reshape(edges.shape[:-1] + (hidden, rank))
    gamma = jax.nn.sigmoid(edges @ params["W_gamma"] + params["b_gamma"])
    scores = (edges @ params["W_a"] + params["b_a"]).squeeze(-1)
    return d, u, v, gamma, scores


def gauge_layer(
    params: dict,
    h: Array,
    edges: Array,
    neighbors: Array,
    *,
    skip_transport: bool = False,
    mask: Array | None = None,
) -> tuple[Array, Array, Array]:
    """One gauge-aware mesh layer: heads, T, attention mix, residual ρ."""
    d, u, v, gamma, scores = transport_heads(params, edges)
    h_j = h[neighbors]
    if skip_transport:
        th = h_j
    else:
        th = apply_transport(d, u, v, gamma, h_j)
    m, alpha = attention_aggregate(th, scores, mask=mask)
    rho = params.get("rho")
    if rho is None:
        rho_fn = None
    else:
        rho_fn = lambda x, _rho=rho: x @ _rho["W"] + _rho["b"]
    h_new = residual_update(h, m, rho=rho_fn)
    return h_new, m, alpha
