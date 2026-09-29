"""Calibrated derivative-process sensitivity (DPS) for GP variable selection."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gp_dps._host import preload as _preload

_preload()

from gp_dps.block import average_block_sensitivities, block_dps
from gp_dps.bootstrap import loo_residuals, multiplier_bootstrap, studentized_pvalues
from gp_dps.gp import fit_se_ard, log_marginal_likelihood, posterior_mean
from gp_dps.kernel import prior_deriv_variance, se_ard, se_ard_deriv_first
from gp_dps.select import bh_reject, by_reject, dps, dps_from_params
from gp_dps.sensitivity import sensitivity_nu

__all__ = [
    "average_block_sensitivities",
    "bh_reject",
    "block_dps",
    "by_reject",
    "dps",
    "dps_from_params",
    "fit_se_ard",
    "log_marginal_likelihood",
    "loo_residuals",
    "multiplier_bootstrap",
    "posterior_mean",
    "prior_deriv_variance",
    "se_ard",
    "se_ard_deriv_first",
    "sensitivity_nu",
    "studentized_pvalues",
]
