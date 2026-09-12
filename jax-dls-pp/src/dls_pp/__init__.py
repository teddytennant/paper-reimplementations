"""Dynamic latent-space inhomogeneous Poisson processes (arXiv:2609.08813)."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from dls_pp.estimate import (
    FitResult,
    activity_block_solve,
    apply_anchor,
    constant_coefficients,
    fit_dls_pp,
    linear_coefficients,
    mean_pairwise_intensity,
    project_coefficients_c0,
    project_positions_c0,
    simulate_network,
)
from dls_pp.intensity import (
    apply_rigid_transform,
    intensity,
    log_intensity,
    pairwise_log_intensity,
    pairwise_squared_distances,
    squared_distance,
)
from dls_pp.likelihood import (
    Events,
    closed_form_constant_loglik,
    dyad_indices,
    homogeneous_loglik,
    poisson_loglik,
    quadrature_grid,
)
from dls_pp.penalties import (
    f1,
    f2,
    is_positive_definite,
    omega_matrices,
    penalty_matrix,
    quadratic_form_penalty,
    roughness_penalty,
)
from dls_pp.splines import (
    SPLINE_DEGREE,
    basis_matrix,
    basis_vector,
    evaluate_trajectories,
    greville_abscissae,
    make_basis,
    squared_distance_quadratic,
)

__all__ = [
    "Events",
    "FitResult",
    "SPLINE_DEGREE",
    "activity_block_solve",
    "apply_anchor",
    "apply_rigid_transform",
    "basis_matrix",
    "basis_vector",
    "closed_form_constant_loglik",
    "constant_coefficients",
    "dyad_indices",
    "evaluate_trajectories",
    "f1",
    "f2",
    "fit_dls_pp",
    "greville_abscissae",
    "homogeneous_loglik",
    "intensity",
    "is_positive_definite",
    "linear_coefficients",
    "log_intensity",
    "make_basis",
    "mean_pairwise_intensity",
    "omega_matrices",
    "pairwise_log_intensity",
    "pairwise_squared_distances",
    "penalty_matrix",
    "poisson_loglik",
    "project_coefficients_c0",
    "project_positions_c0",
    "quadratic_form_penalty",
    "quadrature_grid",
    "roughness_penalty",
    "simulate_network",
    "squared_distance",
    "squared_distance_quadratic",
]
