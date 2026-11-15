"""Weak-form snapshot drift estimator (arXiv:2609.35060)."""

from . import _host

_host.preload()

import jax

jax.config.update("jax_enable_x64", True)

from .estimator import (
    antisym,
    displacement_velocity,
    gaussian_ot_map,
    gradient_only_fit,
    mean_displacement_field,
    prop15_schur_eigs,
    rel_err_solenoidal,
    ridge_fit,
)
from .moments import (
    e_generator_coord,
    e_generator_sqnorm,
    ito_coord_pair,
    ito_sqnorm_pair,
    trapezoidal_residual,
    weak_design_from_samples,
    weak_design_from_second_moments,
)
from .sde import (
    lyapunov_rhs,
    ou_covariance,
    ou_mean,
    planted_curl_H,
    rotation_generator,
    sample_independent_slices,
    snapshot_covariances,
)

__all__ = [
    "antisym",
    "displacement_velocity",
    "e_generator_coord",
    "e_generator_sqnorm",
    "gaussian_ot_map",
    "gradient_only_fit",
    "ito_coord_pair",
    "ito_sqnorm_pair",
    "lyapunov_rhs",
    "mean_displacement_field",
    "ou_covariance",
    "ou_mean",
    "planted_curl_H",
    "prop15_schur_eigs",
    "rel_err_solenoidal",
    "ridge_fit",
    "rotation_generator",
    "sample_independent_slices",
    "snapshot_covariances",
    "trapezoidal_residual",
    "weak_design_from_samples",
    "weak_design_from_second_moments",
]
