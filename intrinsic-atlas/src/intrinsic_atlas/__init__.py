"""Posterior-mean intrinsic atlas sampler (arXiv:2609.30566)."""

from intrinsic_atlas.gaussian import (
    FIG4_LAMBDAS,
    FIG4_RHO_PRODUCTS,
    FIG4_T,
    affine_denoiser,
    clean_deviation,
    gaussian_score,
    make_conditioned_x0_fn,
    make_gaussian_x0_fn,
    make_mixture_x0_fn,
    mixture_score,
    tweedie_x0,
    tweedie_x0_from_eps,
)
from intrinsic_atlas.sampler import (
    ddim_step,
    posterior_mean_step,
    sample_ddim,
    sample_posterior_mean,
    sample_stochastic,
    spread_offset,
    stochastic_step,
)
from intrinsic_atlas.schedules import (
    VPSchedule,
    linear_beta_schedule,
    make_vp_schedule,
    rho_product,
    rho_t,
)

__all__ = [
    "FIG4_LAMBDAS",
    "FIG4_RHO_PRODUCTS",
    "FIG4_T",
    "VPSchedule",
    "affine_denoiser",
    "clean_deviation",
    "ddim_step",
    "gaussian_score",
    "linear_beta_schedule",
    "make_conditioned_x0_fn",
    "make_gaussian_x0_fn",
    "make_mixture_x0_fn",
    "make_vp_schedule",
    "mixture_score",
    "posterior_mean_step",
    "rho_product",
    "rho_t",
    "sample_ddim",
    "sample_posterior_mean",
    "sample_stochastic",
    "spread_offset",
    "stochastic_step",
    "tweedie_x0",
    "tweedie_x0_from_eps",
]
