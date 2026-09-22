"""Proximal simplex allocation."""

from proxvf.alloc import (
    allocate,
    entropy,
    entropy_pcsp,
    hinge_residual,
    quadratic_H_inv_ones,
    quadratic_path,
)

__all__ = [
    "allocate",
    "entropy",
    "entropy_pcsp",
    "hinge_residual",
    "quadratic_H_inv_ones",
    "quadratic_path",
]
