"""QuAKE Kalman corrector (arXiv:2609.21407) in JAX."""

from quake.calibrate import observation_params, process_params, process_residual
from quake.kalman import predict, step, update
from quake.sampler import FilterState, correct_loop, correct_step
from quake.transition import (
    e1,
    lagrange_coefficients,
    lagrange_transition,
    observation_row,
    shift_register,
)

__all__ = [
    "FilterState",
    "correct_loop",
    "correct_step",
    "e1",
    "lagrange_coefficients",
    "lagrange_transition",
    "observation_params",
    "observation_row",
    "predict",
    "process_params",
    "process_residual",
    "shift_register",
    "step",
    "update",
]
