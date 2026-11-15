"""Transformed Gaussian incremental samplers with Hermite control variates."""

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

from tgi_cv.estimator import (
    THETA,
    affine_linear_coefficients,
    degree1_modes,
    g_value,
    hermite_coefficients,
    poisson_eta,
    prune_mask,
    qg_value,
    run_tgi_rwm,
    snis,
    summand,
)
from tgi_cv.flow import (
    affine_apply,
    affine_log_det,
    gaussian_log_prob,
    init_affine,
    log_w,
    log_w_affine,
    log_w_transform,
    lower_triangular,
    objective,
    pushforward_gaussian_log_prob,
    sample_tail,
    tail_penalty,
    train_affine,
)
from tgi_cv.hermite import (
    evaluate_modes,
    hermite_multivariate,
    hermite_probabilists,
    mode_order,
)
from tgi_cv.kernel import (
    acceptance_probability,
    asymptotic_factor,
    beta_from_gamma,
    eigenvalue,
    gi_propose,
    mode_eigenvalues,
    tgi_rwm_step,
)

__all__ = [
    "THETA",
    "acceptance_probability",
    "affine_apply",
    "affine_linear_coefficients",
    "affine_log_det",
    "asymptotic_factor",
    "beta_from_gamma",
    "degree1_modes",
    "eigenvalue",
    "ensure_nix_ld",
    "evaluate_modes",
    "g_value",
    "gaussian_log_prob",
    "gi_propose",
    "hermite_coefficients",
    "hermite_multivariate",
    "hermite_probabilists",
    "init_affine",
    "log_w",
    "log_w_affine",
    "log_w_transform",
    "lower_triangular",
    "mode_eigenvalues",
    "mode_order",
    "objective",
    "poisson_eta",
    "prune_mask",
    "pushforward_gaussian_log_prob",
    "qg_value",
    "run_tgi_rwm",
    "sample_tail",
    "snis",
    "summand",
    "tail_penalty",
    "tgi_rwm_step",
    "train_affine",
]
