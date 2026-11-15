"""Shared array helpers."""

import jax.numpy as jnp


def default_float_dtype():
    """Floating dtype used when the caller does not pass one.

    Follows the JAX x64 flag: float64 when it is enabled, otherwise float32.
    """
    return jnp.asarray(1.0).dtype
