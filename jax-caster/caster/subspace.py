"""The discriminative subspace of eq. (1).

CASTER runs everything downstream in the span of the leading generalized
eigenvectors of the between-class and within-class scatter matrices, that is,
plain LDA:

    S_b u = lambda S_w u,      U_k0 in R^{d x k0},   k0 = min(k_max, C - 1).

The C - 1 cap is the rank of S_b. I solve the generalized problem by whitening
with S_w^{-1/2} and running a symmetric eigh on S_w^{-1/2} S_b S_w^{-1/2},
never a nonsymmetric solver; the whitened matrix is PSD by construction, so
the eigenvalues are real and the eigenvectors are orthonormal.

Two consequences of that construction are used elsewhere: the returned
directions satisfy u^T S_w u = 1, so the projected pooled within-class
covariance Sigma_s of eq. (8) is close to the identity, and the projected
global covariance C_s of eq. (3) is close to diagonal.
"""

from __future__ import annotations

import jax.numpy as jnp

from .linalg import psd_inv_sqrt, symmetrize


def scatter_matrices(features, labels, num_classes):
    """Between-class and within-class scatter, plus the per-class means.

    Both scatters are normalized by the number of samples, which cancels in
    the generalized eigenproblem but keeps the numbers scale-free.

    Returns (S_b, S_w, class_means, counts, global_mean).
    """
    h = jnp.asarray(features)
    y = jnp.asarray(labels)
    n, d = h.shape
    onehot = (y[:, None] == jnp.arange(num_classes)[None, :]).astype(h.dtype)
    counts = onehot.sum(axis=0)
    safe = jnp.maximum(counts, 1.0)
    class_means = (onehot.T @ h) / safe[:, None]
    global_mean = h.mean(axis=0)

    centered_means = class_means - global_mean[None, :]
    s_b = (centered_means * counts[:, None]).T @ centered_means / n

    residuals = h - class_means[y]
    s_w = residuals.T @ residuals / n
    return symmetrize(s_b), symmetrize(s_w), class_means, counts, global_mean


def lda_directions(s_b, s_w, k0, within_ridge):
    """Leading k0 generalized eigenvectors of (S_b, S_w), eq. (1).

    `within_ridge` is relative: the ridge added to S_w is
    within_ridge * trace(S_w) / d.
    """
    d = s_w.shape[0]
    scale = jnp.trace(s_w) / d
    reg = s_w + jnp.asarray(within_ridge, s_w.dtype) * scale * jnp.eye(d, dtype=s_w.dtype)
    whitener = psd_inv_sqrt(reg)
    whitened = symmetrize(whitener @ s_b @ whitener)
    eigvals, eigvecs = jnp.linalg.eigh(whitened)
    # eigh returns ascending order; take the k0 largest.
    top = eigvecs[:, ::-1][:, :k0]
    directions = whitener @ top
    # Fix the sign per column so repeated fits on the same data agree. Nothing
    # downstream depends on it: flipping a column flips one coordinate of z and
    # the matching coordinate of every stored moment.
    pivot = jnp.take_along_axis(
        directions, jnp.argmax(jnp.abs(directions), axis=0)[None, :], axis=0
    )[0]
    return directions * jnp.sign(jnp.where(pivot == 0, 1.0, pivot))[None, :]


def subspace_dim(num_classes, k_max, feature_dim):
    """k0 = min(k_max, C - 1) from eq. (1), also capped by the feature dimension."""
    return int(min(k_max, num_classes - 1, feature_dim))


def active_dim(k0, batch_size):
    """k = min(k0, B - 1) from eq. (2)."""
    return int(max(1, min(k0, batch_size - 1)))
