"""SME-BETEL: score matching first-order conditions inside Bayesian ETEL.

An implementation of arXiv:2609.01783, "Robust Bayesian Inference for
Unnormalized Models with Mixed-Domain Data".
"""

import jax

# Hessians in x, a Newton solve for the tilt, and matrix inverses in the
# sandwich covariance. Single precision does not hold any of these together.
jax.config.update("jax_enable_x64", True)

from .asymptotics import delta_matrix, gamma_matrix, sandwich_covariance
from .etel import ETELSolution, etel_weights, log_etel, solve_tilt
from .mixed import (
    boundary_weight_1d,
    mixed_estimator,
    box_weight,
    mixed_integrand,
    mixed_moment_fn,
    mixed_objective,
    smoothstep,
)
from .posterior import MHOutput, log_posterior_fn, metropolis_hastings
from .scorematch import (
    moment_fn,
    moment_matrix,
    newton_minimize,
    score_integrand,
    sm_estimator,
    sm_objective,
)

__all__ = [
    "ETELSolution",
    "MHOutput",
    "boundary_weight_1d",
    "box_weight",
    "delta_matrix",
    "etel_weights",
    "gamma_matrix",
    "log_etel",
    "log_posterior_fn",
    "metropolis_hastings",
    "mixed_integrand",
    "mixed_estimator",
    "mixed_moment_fn",
    "mixed_objective",
    "moment_fn",
    "moment_matrix",
    "newton_minimize",
    "sandwich_covariance",
    "score_integrand",
    "sm_estimator",
    "sm_objective",
    "smoothstep",
    "solve_tilt",
]
