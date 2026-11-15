"""The class-shared affine map of eq. (3) and the transported moments of eq. (4).

    A_t = (Chat_t + lambda I)^{1/2} (C_s + lambda I)^{-1/2}
    b_t = mhat_t - A_t m_s

with symmetric PSD roots. `A_t` always satisfies

    A_t (C_s + lambda I) A_t^T = Chat_t + lambda I

whatever lambda is, and when C_s = I and the true shift is a symmetric
positive definite A_true, the small-lambda limit of A_t is A_true exactly. Both
are checked in tests/test_transport.py.

Sec. 3.1's sample-efficiency safeguard: use the dense map when B >= 2k, and the
diagonal analogue (both covariances replaced by their diagonal variances) when
B < 2k. The dense map has O(k^2) degrees of freedom against B samples; the
diagonal one has O(k).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .linalg import psd_inv_sqrt, psd_sqrt, ridge, symmetrize


class AffineMap(NamedTuple):
    """The estimated transport, plus which form of eq. (3) produced it."""

    a: jnp.ndarray      # A_t, (k, k)
    b: jnp.ndarray      # b_t, (k,)
    diagonal: bool      # True if the B < 2k diagonal analogue was used


def empirical_moments(batch):
    """mhat_t and Chat_t for a target batch, with the n - 1 denominator."""
    z = jnp.asarray(batch)
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    return mean, symmetrize(cov)


def use_diagonal_form(batch_size, k):
    """Sec. 3.1's switch: diagonal when B < 2k, dense otherwise."""
    return batch_size < 2 * k


def whiten_color_map(source_mean, source_cov, batch_mean, batch_cov, lam, diagonal):
    """Equation (3), from moments rather than raw samples."""
    m_s = jnp.asarray(source_mean)
    c_s = jnp.asarray(source_cov)
    lam = jnp.asarray(lam, c_s.dtype)

    if diagonal:
        var_t = jnp.diagonal(batch_cov) + lam
        var_s = jnp.diagonal(c_s) + lam
        a = jnp.diag(jnp.sqrt(jnp.clip(var_t, 0.0, None) / var_s))
    else:
        a = psd_sqrt(ridge(batch_cov, lam)) @ psd_inv_sqrt(ridge(c_s, lam))

    return AffineMap(a=a, b=batch_mean - a @ m_s, diagonal=bool(diagonal))


def estimate_transport(source_mean, source_cov, batch, lam, force_form=None):
    """Equation (3) for a target batch, choosing the dense or diagonal form.

    `force_form` is "full" or "diag" to override the B vs 2k rule; the default
    None follows Sec. 3.1.
    """
    z = jnp.asarray(batch)
    k = z.shape[1]
    batch_mean, batch_cov = empirical_moments(z)
    if force_form is None:
        diagonal = use_diagonal_form(z.shape[0], k)
    elif force_form in ("diag", "diagonal"):
        diagonal = True
    elif force_form == "full":
        diagonal = False
    else:
        raise ValueError(f"force_form must be None, 'full' or 'diag', got {force_form!r}")
    return whiten_color_map(source_mean, source_cov, batch_mean, batch_cov, lam, diagonal)


def transport_class_moments(affine: AffineMap, means, covs, beta):
    """Equation (4): mu~ = A mu + b and Sigma~ = A Sigma A^T + beta I."""
    means = jnp.asarray(means)
    covs = jnp.asarray(covs)
    transported_means = means @ affine.a.T + affine.b[None, :]
    transported_covs = affine.a[None] @ covs @ affine.a.T[None]
    return transported_means, ridge(symmetrize(transported_covs), beta)


def inverse_transport(affine: AffineMap, z):
    """Equation (6): zbar = A^-1 (z - b), solved rather than inverted."""
    return jnp.linalg.solve(affine.a, (jnp.asarray(z) - affine.b[None, :]).T).T
