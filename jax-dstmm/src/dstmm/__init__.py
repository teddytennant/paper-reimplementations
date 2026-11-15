"""JAX implementation of the deep skew-t mixture model (DStMM), arXiv:2609.00773.

Everything in this package runs in float64. The model needs matrix inverses,
log-determinants, Bessel functions of the third kind and log-likelihood
differences, and float32 is not enough for any of them. On NVIDIA hardware
float32 matmuls also default to TF32, which silently costs another two digits,
so x64 is switched on here at import time before any array is created.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

if _jax.numpy.zeros(1).dtype != _jax.numpy.float64:  # pragma: no cover
    raise RuntimeError("dstmm needs float64; jax_enable_x64 did not take effect")

from dstmm.bessel import log_bessel_k, dlog_bessel_k_dnu  # noqa: E402
from dstmm.gig import (  # noqa: E402
    gig_logpdf,
    gig_log_norm,
    gig_moment,
    gig_log_moment,
    sample_gig,
    inverse_gamma_logpdf,
    sample_inverse_gamma,
)
from dstmm.ghst import (  # noqa: E402
    ghst_logpdf,
    ghst_mean,
    ghst_cov,
    sample_ghst,
    student_t_logpdf,
    posterior_w_params,
)
from dstmm.model import (  # noqa: E402
    DStMM,
    collapse_paths,
    init_from_data,
    layer_responsibilities,
    log_likelihood,
    path_log_densities,
    path_responsibilities,
    per_point_log_likelihood,
    n_params,
    bic,
    sample_dstmm,
    random_init,
)
from dstmm.mcem import MCEMResult, mcem_fit, mcem_step  # noqa: E402
from dstmm.simulate import paper_style_truth  # noqa: E402

__all__ = [
    "log_bessel_k",
    "dlog_bessel_k_dnu",
    "gig_logpdf",
    "gig_log_norm",
    "gig_moment",
    "gig_log_moment",
    "sample_gig",
    "inverse_gamma_logpdf",
    "sample_inverse_gamma",
    "ghst_logpdf",
    "ghst_mean",
    "ghst_cov",
    "sample_ghst",
    "student_t_logpdf",
    "posterior_w_params",
    "DStMM",
    "collapse_paths",
    "init_from_data",
    "layer_responsibilities",
    "log_likelihood",
    "path_log_densities",
    "path_responsibilities",
    "per_point_log_likelihood",
    "n_params",
    "bic",
    "sample_dstmm",
    "random_init",
    "mcem_fit",
    "mcem_step",
    "MCEMResult",
    "paper_style_truth",
]
