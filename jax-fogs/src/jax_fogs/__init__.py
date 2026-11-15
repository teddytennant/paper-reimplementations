"""FO-GS: fractional-order OLS grid-search (arXiv:2609.18127)."""

import jax

jax.config.update("jax_enable_x64", True)

from jax_fogs.estimate import alpha_grid, fogs, fogs_commensurate
from jax_fogs.operators import (
    fractional_difference,
    gl_psi,
    green_matrices,
    simulate_folti,
)

__version__ = "0.1.0"

__all__ = [
    "alpha_grid",
    "fogs",
    "fogs_commensurate",
    "fractional_difference",
    "gl_psi",
    "green_matrices",
    "simulate_folti",
]
