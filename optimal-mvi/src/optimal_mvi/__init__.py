"""Unconstrained p=2 optimal high-order method for monotone variational inequalities."""

import jax

jax.config.update("jax_enable_x64", True)

from optimal_mvi.halpern import inexact_accelerated_halpern, parameter_schedule
from optimal_mvi.tensor import (
    A_P,
    M_P,
    THETA_P,
    anchored_tensor_method,
    r_adaptive_atm,
    tensor_step,
)

__all__ = [
    "A_P",
    "M_P",
    "THETA_P",
    "anchored_tensor_method",
    "inexact_accelerated_halpern",
    "parameter_schedule",
    "r_adaptive_atm",
    "tensor_step",
]
