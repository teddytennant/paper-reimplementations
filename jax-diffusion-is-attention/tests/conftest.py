"""Run the whole suite in float64 so the tolerances mean something."""

import jax

jax.config.update("jax_enable_x64", True)
