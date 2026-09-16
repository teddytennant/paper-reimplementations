"""Tiny synthetic classification sets. No MNIST/CIFAR downloads."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gaussian_blobs(
    key: jax.Array,
    n: int = 64,
    n_classes: int = 4,
    dim: int = 2,
    std: float = 0.25,
    scale: float = 2.0,
) -> tuple[jax.Array, jax.Array]:
    """Well-separated Gaussian blobs, linearly separable for small std."""
    # Place class means on a circle (2D) or on scaled one-hot axes.
    if dim == 2:
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_classes, endpoint=False)
        centers = scale * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    else:
        centers = scale * jnp.eye(n_classes, dim)

    y = jnp.repeat(jnp.arange(n_classes), (n + n_classes - 1) // n_classes)[:n]
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, (n, dim)) * std
    x = centers[y] + noise
    y_oh = jax.nn.one_hot(y, n_classes)
    return x, y_oh


def linearly_separable_batch(n: int = 32, dim: int = 2) -> tuple[jax.Array, jax.Array]:
    """Two-class points on opposite sides of a hyperplane."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, dim))
    x = x.at[:, 0].set(jnp.where(jnp.arange(n) < n // 2, -1.5, 1.5))
    y_idx = (jnp.arange(n) >= n // 2).astype(jnp.int32)
    y = jax.nn.one_hot(y_idx, 2)
    return x, y
