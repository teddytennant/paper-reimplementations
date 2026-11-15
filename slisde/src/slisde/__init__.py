"""SLiSDE: structured linear SDE layers with gated in-flow stacking."""

from . import _host

_host.preload()

from .gates import apply_inflow, gated_pairs, scale_and_offset, scale_diag, zero_gates
from .girsanov import log_dp_over_dq, log_dq_over_dp, snis_expect, snis_weights
from .loss import calibration_loss
from .model import init_model, simulate
from .scan import affine_compose, parallel_simulate, sequential_simulate
from .sde import euler_maruyama_pairs, exponential_pairs, transition_pairs

__all__ = [
    "affine_compose",
    "apply_inflow",
    "calibration_loss",
    "euler_maruyama_pairs",
    "exponential_pairs",
    "gated_pairs",
    "init_model",
    "log_dp_over_dq",
    "log_dq_over_dp",
    "parallel_simulate",
    "scale_and_offset",
    "scale_diag",
    "sequential_simulate",
    "simulate",
    "snis_expect",
    "snis_weights",
    "transition_pairs",
    "zero_gates",
]
