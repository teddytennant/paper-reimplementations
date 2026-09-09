"""Kernels and MMD helpers for arXiv:2609.07432.

Gaussian target kernel and square-root kernel follow Definition 1:
  k(x, y) = exp(-||x-y||^2 / (2 sigma^2))
  k_rt(x, y) = (2 / (pi sigma^2))^{d/4} exp(-||x-y||^2 / sigma^2)
so that k(x, y) = integral k_rt(x, z) k_rt(y, z) dz.
"""

from __future__ import annotations

import jax.numpy as jnp


def pairwise_sqdist(x, y=None):
    """Squared Euclidean distances via explicit coordinate sums.

    Args:
        x: (n, d) points.
        y: (m, d) points, or None to use x.

    Returns:
        (n, m) matrix of ||x_i - y_j||^2.
    """
    if y is None:
        y = x
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def gaussian_kernel(x, y=None, sigma: float = 1.0):
    """Gaussian kernel k(x, y) = exp(-||x-y||^2 / (2 sigma^2))."""
    sq = pairwise_sqdist(x, y)
    return jnp.exp(-sq / (2.0 * sigma * sigma))


def gaussian_sqrt_kernel(x, y=None, sigma: float = 1.0):
    """Square-root kernel k_rt of the Gaussian kernel (Definition 1).

    k_rt(x, y) = (2 / (pi sigma^2))^{d/4} exp(-||x-y||^2 / sigma^2).
    """
    d = x.shape[-1]
    const = (2.0 / (jnp.pi * sigma * sigma)) ** (d / 4.0)
    sq = pairwise_sqdist(x, y)
    return const * jnp.exp(-sq / (sigma * sigma))


def linear_kernel_matrix(x, y=None):
    """Linear kernel k(x, y) = x · y, matching explicit features Phi(x) = x."""
    if y is None:
        y = x
    return x @ y.T


def mmd2(k_xx, k_yy, k_xy):
    """Squared MMD between two empirical measures from kernel blocks.

    MMD^2(P, Q) = mean k(x, x') + mean k(y, y') - 2 mean k(x, y).
    """
    return jnp.mean(k_xx) + jnp.mean(k_yy) - 2.0 * jnp.mean(k_xy)


def mmd2_from_points(x, y, kernel_fn):
    """Squared empirical MMD between point sets x and y."""
    return mmd2(kernel_fn(x, x), kernel_fn(y, y), kernel_fn(x, y))


def mmd2_features(phi_a, phi_b):
    """Squared MMD with explicit features: ||mean(phi_a) - mean(phi_b)||^2."""
    mu_a = jnp.mean(phi_a, axis=0)
    mu_b = jnp.mean(phi_b, axis=0)
    diff = mu_a - mu_b
    return jnp.sum(diff * diff)


def kernel_submatrix(k, idx_a, idx_b=None):
    """Rows idx_a and columns idx_b of a kernel matrix."""
    idx_a = jnp.asarray(idx_a)
    if idx_b is None:
        idx_b = idx_a
    else:
        idx_b = jnp.asarray(idx_b)
    return k[idx_a[:, None], idx_b[None, :]]


def mmd2_indices(k, idx_a, idx_b):
    """Squared empirical MMD between two index sets given kernel matrix k."""
    return mmd2(
        kernel_submatrix(k, idx_a, idx_a),
        kernel_submatrix(k, idx_b, idx_b),
        kernel_submatrix(k, idx_a, idx_b),
    )
