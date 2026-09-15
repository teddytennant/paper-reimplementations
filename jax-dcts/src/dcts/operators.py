"""Linear forward operators and the covariance-scaled update (Eq. 10).

    m^y = m + c A^T (c A A^T + Sigma_y)^{-1} (y - A m)

with Sigma_y = sigma_y^2 I. Tiny synthetic problems use dense A.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


def gaussian_matrix(key: Array, m: int, d: int, scale: float = 1.0) -> Array:
    """Gaussian sensing matrix with columns of expected norm `scale`."""
    a = jax.random.normal(key, (m, d))
    return a * (scale / jnp.sqrt(d))


def inpainting_matrix(keep_mask: Array) -> Array:
    """Row-subsampled identity: y = x[keep]."""
    mask = np.asarray(keep_mask).astype(bool)
    keep_idx = np.flatnonzero(mask)
    d = mask.shape[0]
    return jnp.eye(d, dtype=jnp.float32)[keep_idx]


def box_downsample_matrix(height: int, width: int, factor: int, dtype=jnp.float32) -> Array:
    """Disjoint box averaging, row-major images flattened to vectors."""
    if height % factor != 0 or width % factor != 0:
        raise ValueError("height and width must be divisible by factor")
    out_h = height // factor
    out_w = width // factor
    m = out_h * out_w
    d = height * width
    a = np.zeros((m, d), dtype=np.float32)
    weight = 1.0 / (factor * factor)
    for i in range(out_h):
        for j in range(out_w):
            row = i * out_w + j
            for di in range(factor):
                for dj in range(factor):
                    r = i * factor + di
                    c = j * factor + dj
                    a[row, r * width + c] = weight
    return jnp.asarray(a, dtype=dtype)


def apply(a: Array, x: Array) -> Array:
    return a @ x


def apply_t(a: Array, y: Array) -> Array:
    return a.T @ y


def measurement_update(
    mean: Array,
    cov_scale: Array,
    y: Array,
    a: Array,
    sigma_y: Array,
) -> Array:
    """Eq. 10 for y = A x + N(0, sigma_y^2 I).

    residual r = A m - y is formed internally as y - A m.
    g = A^T (c A A^T + Sigma_y)^{-1} (y - A m), then m^y = m + c g.
    """
    residual = y - a @ mean
    m_obs = a.shape[0]
    gram = a @ a.T
    eye = jnp.eye(m_obs, dtype=mean.dtype)
    system = cov_scale * gram + (sigma_y**2) * eye
    u = jnp.linalg.solve(system, residual)
    return mean + cov_scale * (a.T @ u)
