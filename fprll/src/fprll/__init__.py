"""Restarted follow-the-perturbed-regularized-lazy-leader on the box."""

import jax

jax.config.update("jax_enable_x64", True)

from fprll.aggregate import Combine, DMMultiscale, FixedShare
from fprll.core import (
    GRID_N,
    decision_bin,
    density,
    indicator_cost,
    lazy_sample,
    make_grid,
    minimizer,
    potential_grad,
    potential_hess,
    total_variation,
)
from fprll.simulate import FPRLL, RestartedFPRLL, dyadic_scales, rollout_scale, run

__all__ = [
    "GRID_N",
    "Combine",
    "DMMultiscale",
    "FPRLL",
    "FixedShare",
    "RestartedFPRLL",
    "decision_bin",
    "density",
    "dyadic_scales",
    "indicator_cost",
    "lazy_sample",
    "make_grid",
    "minimizer",
    "potential_grad",
    "potential_hess",
    "rollout_scale",
    "run",
    "total_variation",
]
