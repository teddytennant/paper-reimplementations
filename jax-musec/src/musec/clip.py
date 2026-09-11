"""Spectral clipping maps from arXiv:2609.11655.

Hard clip: Section 3.1 and Algorithm 2 (steps 11–13)
    clip(w, D) = min{w, D}
    M = U Clip(S, D) V^T

Soft clip: Section 5, Eq. (3) and Algorithm 3
    h(w, D) = D w / sqrt(w^2 + D^2)
    H(M, D) = D (M M^T + D^2 I)^{-1/2} M
    inverse square root via coupled Newton–Schulz (Higham 2008, Ch. 6)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def clip_singular_values(s: Array, D: float) -> Array:
    """Entrywise clip(w, D) = min{w, D} (Section 3.1)."""
    return jnp.minimum(s, D)


def soft_saturation(s: Array, D: float) -> Array:
    """Smooth saturation h(w, D) = D w / sqrt(w^2 + D^2) (Section 5)."""
    D = jnp.asarray(D, dtype=s.dtype)
    return D * s / jnp.sqrt(s * s + D * D)


def _svd_spectral_map(M: Array, s_fn) -> Array:
    """Apply a scalar map to singular values, keep U, V from reduced SVD."""
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    s_new = s_fn(s)
    return (U * s_new) @ Vh


def hard_spectral_clip(M: Array, D: float) -> Array:
    """Exact Musec clip: M |-> U Clip(S, D) V^T (Algorithm 2, steps 11–13)."""
    return _svd_spectral_map(M, lambda s: clip_singular_values(s, D))


# Alias used in the optimizer.
spectral_clip = hard_spectral_clip


def soft_spectral_clip_exact(M: Array, D: float) -> Array:
    """Exact soft map H(M, D) via SVD (Section 5, Eq. (3))."""
    return _svd_spectral_map(M, lambda s: soft_saturation(s, D))


def soft_spectral_clip(M: Array, D: float, ns_steps: int = 5) -> Array:
    """Soft spectral clipping via coupled Newton–Schulz (Algorithm 3).

    Algorithm 3 assumes m <= n without loss of generality. If M has more
    rows than columns we apply SSC to M^T and transpose the result, which
    is equivalent to H(M, D) = D M (M^T M + D^2 I)^{-1/2}.

    The NS coefficients are Higham's inverse-square-root iteration
        T_k = (1/2) (3 I - Z_k Y_k),
    not Muon's quintic polar polynomial.
    """
    m, n = M.shape
    transposed = m > n
    M_work = M.T if transposed else M
    out = _ssc_small_first(M_work, D, ns_steps)
    return out.T if transposed else out


def _ssc_small_first(M: Array, D: float, ns_steps: int) -> Array:
    """Algorithm 3 on M with m <= n: return D (Z_K / sqrt(alpha)) M."""
    m = M.shape[0]
    dtype = M.dtype
    D2 = jnp.asarray(D, dtype=dtype) ** 2
    A = M @ M.T + D2 * jnp.eye(m, dtype=dtype)
    alpha = jnp.linalg.norm(A, ord="fro")
    # alpha >= m * D^2 > 0 for D > 0.
    Y = A / alpha
    Z = jnp.eye(m, dtype=dtype)
    I = jnp.eye(m, dtype=dtype)

    def body(_, carry):
        Y, Z = carry
        T = 0.5 * (3.0 * I - Z @ Y)
        return Y @ T, T @ Z

    Y, Z = jax.lax.fori_loop(0, ns_steps, body, (Y, Z))
    inv_sqrt_A = Z / jnp.sqrt(alpha)
    return jnp.asarray(D, dtype=dtype) * (inv_sqrt_A @ M)
