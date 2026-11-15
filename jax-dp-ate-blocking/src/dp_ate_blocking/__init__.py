"""JAX implementation of DP ATE by propensity score blocking (arXiv:2609.09536)."""

from .dgp import well_specified
from .estimators import (
    aipw,
    aipw_sensitivity,
    assign_bins,
    bps,
    clip_propensity,
    difference_in_means,
    ipw,
    ipw_sensitivity,
)
from .privacy import default_m, dp_blocking, sample_laplace, seq_ipw
from .propensity import fit_logistic_l2, fit_propensity, predict_propensity

__all__ = [
    "aipw",
    "aipw_sensitivity",
    "assign_bins",
    "bps",
    "clip_propensity",
    "default_m",
    "difference_in_means",
    "dp_blocking",
    "fit_logistic_l2",
    "fit_propensity",
    "ipw",
    "ipw_sensitivity",
    "predict_propensity",
    "sample_laplace",
    "seq_ipw",
    "well_specified",
]

__version__ = "0.1.0"
