"""JAX kernel of MSVR for finite-sum coupled compositional optimization."""

from jax_msvr.estimator import (
    correction_gamma,
    finite_sum_g_hat,
    msvr_fs_update,
    msvr_sp_update,
    msvr_update,
    sox_update,
)
from jax_msvr.metrics import objective, tracking_error
from jax_msvr.msvrm import (
    MSVRMState,
    init_msvrm_state,
    msvrm_v1_step,
    msvrm_v2_step,
    msvrm_v3_step,
)
from jax_msvr.problems import QuadraticFCCO, TinyAUC, make_quadratic
from jax_msvr.sox import sox_step

__all__ = [
    "correction_gamma",
    "finite_sum_g_hat",
    "msvr_fs_update",
    "msvr_sp_update",
    "msvr_update",
    "sox_update",
    "objective",
    "tracking_error",
    "MSVRMState",
    "init_msvrm_state",
    "msvrm_v1_step",
    "msvrm_v2_step",
    "msvrm_v3_step",
    "QuadraticFCCO",
    "TinyAUC",
    "make_quadratic",
    "sox_step",
]
