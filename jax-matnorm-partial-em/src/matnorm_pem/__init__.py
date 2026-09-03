"""Partial EM for matrix-variate normal models with missing data.

Implements the algorithms in "An efficient EM algorithm for both element-wise
and structural missingness in matrix-variate normal mixture models"
(Lu, Andrews and Browne, arXiv:2609.00616v1).

Everything runs in float64. Matrix normal EM inverts covariance factors and
compares log-likelihoods across iterations, and float32 (worse, TF32 matmuls on
NVIDIA) loses enough precision to hide real bugs, so x64 is switched on here at
import time before any array is built.
"""

import jax

jax.config.update("jax_enable_x64", True)

from .params import MatNormParams, normalize_scale  # noqa: E402
from .density import (  # noqa: E402
    vec,
    unvec,
    kron_cov,
    matnorm_logpdf,
    mvn_logpdf,
    sample_matnorm,
)
from .estep import (  # noqa: E402
    exact_estep,
    exact_estep_by_pattern,
    partial_estep,
    plugin_estep,
    observed_loglik,
    free_energy,
    init_estep_state,
    EStepState,
)
from .mstep import row_correction, col_correction, mstep  # noqa: E402
from .em import fit_matnorm  # noqa: E402
from .structural import (  # noqa: E402
    block_mask,
    structural_estep,
    structural_corrections,
    fit_matnorm_structural,
)
from .mixture import fit_mixture, MixtureParams  # noqa: E402
from . import opcount, synth  # noqa: E402

__all__ = [
    "MatNormParams",
    "MixtureParams",
    "EStepState",
    "normalize_scale",
    "vec",
    "unvec",
    "kron_cov",
    "matnorm_logpdf",
    "mvn_logpdf",
    "sample_matnorm",
    "exact_estep",
    "exact_estep_by_pattern",
    "partial_estep",
    "plugin_estep",
    "observed_loglik",
    "free_energy",
    "init_estep_state",
    "row_correction",
    "col_correction",
    "mstep",
    "fit_matnorm",
    "block_mask",
    "structural_estep",
    "structural_corrections",
    "fit_matnorm_structural",
    "fit_mixture",
    "opcount",
    "synth",
]
