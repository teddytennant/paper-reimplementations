"""Backward Kernel Herding and Flexible Kernel Thinning (arXiv:2609.07432)."""

from bkh.bkh import (
    backward_kernel_herding,
    g_remaining_mean,
    j_removal,
    kernelized_backward_herding,
)
from bkh.fkt import dyadic_bits, flexible_kernel_thinning, kt_split_tree
from bkh.kernels import (
    gaussian_kernel,
    gaussian_sqrt_kernel,
    linear_kernel_matrix,
    mmd2,
    mmd2_features,
    pairwise_sqdist,
)

__all__ = [
    "backward_kernel_herding",
    "dyadic_bits",
    "flexible_kernel_thinning",
    "g_remaining_mean",
    "gaussian_kernel",
    "gaussian_sqrt_kernel",
    "j_removal",
    "kernelized_backward_herding",
    "kt_split_tree",
    "linear_kernel_matrix",
    "mmd2",
    "mmd2_features",
    "pairwise_sqdist",
]
