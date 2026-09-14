"""Convex interpolant, one-hot targets, and min-index argmax (eq 8, 16-18)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def convex_interpolant(z0, z1, k_tilde):
    """Z_t = (1 - k_tilde) Z_0 + k_tilde Z_1, with k_tilde broadcast on the left."""
    k = jnp.asarray(k_tilde, dtype=z0.dtype)
    extra = z0.ndim - k.ndim
    if extra < 0:
        raise ValueError("k_tilde has more axes than z0")
    k = jnp.reshape(k, k.shape + (1,) * extra)
    return (1.0 - k) * z0 + k * z1


def argmax_tokens(z, axis=-1):
    """Position-wise argmax. Ties take the smallest index (eq 16-17).

    ``jnp.argmax`` already returns the first maximizer along ``axis``.
    """
    return jnp.argmax(z, axis=axis)


def one_hot_tokens(tokens, vocab_size, dtype=jnp.float32):
    """One-hot encode tokens in {0, ..., V-1} along a new last axis."""
    return jax.nn.one_hot(tokens, vocab_size, dtype=dtype)


def sample_distinct_token(key, x0, vocab_size):
    """Sample x1 uniformly from {0, ..., V-1} \\ {x0}, vectorized."""
    u = jax.random.randint(key, jnp.shape(x0), minval=0, maxval=vocab_size - 1)
    return jnp.where(u >= x0, u + 1, u)
