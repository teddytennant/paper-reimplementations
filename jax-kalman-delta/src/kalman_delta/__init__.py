"""Kalman Delta Network recurrences in JAX (arXiv:2609.07816)."""

from kalman_delta.delta import delta_rule_step, isotropic_gain_scale, kappa_from_beta
from kalman_delta.diagonal import (
    diagonal_gain,
    diagonal_step,
    mobius_matrices,
    mobius_scan,
    variational_diagonal_p,
)
from kalman_delta.isotropic import isotropic_step
from kalman_delta.kalman import kalman_filter, kalman_predict, kalman_step, kalman_update, read

__all__ = [
    "delta_rule_step",
    "diagonal_gain",
    "diagonal_step",
    "isotropic_gain_scale",
    "isotropic_step",
    "kalman_filter",
    "kalman_predict",
    "kalman_step",
    "kalman_update",
    "kappa_from_beta",
    "mobius_matrices",
    "mobius_scan",
    "read",
    "variational_diagonal_p",
]
