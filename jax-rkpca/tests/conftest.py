"""Enable float64 for tighter orthonormality and determinant checks."""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
