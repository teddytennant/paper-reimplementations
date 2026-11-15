"""Symmetric matrix functions used throughout CASTER.

Everything here goes through `jnp.linalg.eigh` with clamped eigenvalues rather
than a generic solver. Equation (3) of the paper asks for "the symmetric
positive-semidefinite roots", and the Mahalanobis forms in equations (8) and
(9) are differences evaluated by cancellation, so a stable symmetric
factorization matters more than speed here. The matrices are k x k with k at
most a few hundred, so the eigendecompositions are cheap.

Note on backends: eigenvector signs and the ordering of degenerate eigenvalues
differ between CPU and GPU. Nothing in this module returns eigenvectors; the
callers only ever see the invariant quantities (the reconstructed matrix
function, a quadratic form, a log-determinant), which agree across backends.
"""

from __future__ import annotations

import jax.numpy as jnp


def symmetrize(m):
    """Average a matrix with its transpose."""
    return 0.5 * (m + jnp.swapaxes(m, -1, -2))


def _eigen_floor(w):
    """Relative floor below which an eigenvalue is treated as zero."""
    eps = jnp.finfo(w.dtype).eps
    scale = jnp.maximum(jnp.max(jnp.abs(w), axis=-1, keepdims=True), 1.0)
    return eps * scale * w.shape[-1]


def psd_sqrt(m):
    """Symmetric PSD square root, negative eigenvalues clamped to zero."""
    w, v = jnp.linalg.eigh(symmetrize(m))
    w = jnp.clip(w, 0.0, None)
    return symmetrize((v * jnp.sqrt(w)[..., None, :]) @ jnp.swapaxes(v, -1, -2))


def psd_inv_sqrt(m):
    """Symmetric PSD inverse square root.

    Eigenvalues below a relative floor are dropped (their inverse root is set
    to zero) instead of blowing up. Callers add a positive ridge first, so in
    practice the floor never fires.
    """
    w, v = jnp.linalg.eigh(symmetrize(m))
    floor = _eigen_floor(w)
    inv_root = jnp.where(w > floor, 1.0 / jnp.sqrt(jnp.where(w > floor, w, 1.0)), 0.0)
    return symmetrize((v * inv_root[..., None, :]) @ jnp.swapaxes(v, -1, -2))


def psd_logdet_and_quadform(m, residuals):
    """Return (log det m, r^T m^-1 r) for a batch of residuals.

    `m` is (..., k, k) symmetric positive definite and `residuals` is
    (..., k). Used for equation (5), where the same factorization gives both
    terms.
    """
    w, v = jnp.linalg.eigh(symmetrize(m))
    floor = _eigen_floor(w)
    w = jnp.maximum(w, floor)
    projected = jnp.einsum("...ij,...i->...j", v, residuals)
    quad = jnp.sum(projected * projected / w, axis=-1)
    return jnp.sum(jnp.log(w), axis=-1), quad


def mahalanobis_sq(metric, vectors):
    """v^T metric^-1 v for one or more vectors sharing a single metric."""
    w, v = jnp.linalg.eigh(symmetrize(metric))
    floor = _eigen_floor(w)
    w = jnp.maximum(w, floor)
    projected = jnp.asarray(vectors) @ v
    return jnp.sum(projected * projected / w, axis=-1)


def ridge(m, amount):
    """m + amount * I, with the identity in m's dtype."""
    k = m.shape[-1]
    return m + jnp.asarray(amount, m.dtype) * jnp.eye(k, dtype=m.dtype)
