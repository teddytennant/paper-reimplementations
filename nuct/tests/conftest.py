"""Enable float64 before the package, or any test, builds an array."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
