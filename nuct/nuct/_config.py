"""Float64 before any ``jax.numpy`` import in this package.

arXiv:2609.40277 is checked in complex128 / float64. JAX defaults to
float32, which is not enough for the 1e-12 identity tests.
"""

from __future__ import annotations

import jax

def ensure_x64() -> None:
    """Turn on JAX float64. Idempotent; safe to call more than once."""
    jax.config.update("jax_enable_x64", True)


ensure_x64()
