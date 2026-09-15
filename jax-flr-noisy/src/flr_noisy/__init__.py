"""Two-step Fourier FLR estimator for noisy grid covariates (arXiv:2609.08671)."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from flr_noisy.basis import (
    ALPHA,
    KAPPA,
    dimension,
    discrete_inner_product,
    eigenvalue_threshold,
    evaluate_fourier,
    fourier_basis_at,
    fourier_basis_matrix,
    lemma_a1_inner_product,
    model_index,
    penalty,
    regular_grid,
    theoretical_max_model_index,
    theoretical_recon_dim,
)
from flr_noisy.estimate import (
    FitResult,
    fit_fixed_m,
    fit_flr,
    gram_and_rhs,
    predict,
    predict_from_grid,
    select_model,
)
from flr_noisy.reconstruct import (
    fourier_coefficients,
    reconstruct,
    reconstruct_on_grid,
)
from flr_noisy.simulate import (
    SyntheticData,
    decaying_beta_coeffs,
    generate_kl_covariates,
    generate_synthetic,
    kl_eigenvalues,
    quadratic_beta_coeffs,
)

__all__ = [
    "ALPHA",
    "KAPPA",
    "FitResult",
    "SyntheticData",
    "decaying_beta_coeffs",
    "dimension",
    "discrete_inner_product",
    "eigenvalue_threshold",
    "evaluate_fourier",
    "fit_fixed_m",
    "fit_flr",
    "fourier_basis_at",
    "fourier_basis_matrix",
    "fourier_coefficients",
    "generate_kl_covariates",
    "generate_synthetic",
    "gram_and_rhs",
    "kl_eigenvalues",
    "lemma_a1_inner_product",
    "model_index",
    "penalty",
    "predict",
    "predict_from_grid",
    "quadratic_beta_coeffs",
    "reconstruct",
    "reconstruct_on_grid",
    "regular_grid",
    "select_model",
    "theoretical_max_model_index",
    "theoretical_recon_dim",
]
