"""Distributed kernel-based robust gradient descent (DKRGD), arXiv:2609.11712."""

from dkrgd.distributed import LocalModel, fit_dkrgd, partition, predict_dkrgd
from dkrgd.kernel import gaussian_kernel, kernel_matrix
from dkrgd.krgd import fit_krgd, predict_krgd, step_size_bound
from dkrgd.losses import G, G_prime, G_prime_plus_0, influence_weight, l_sigma

__all__ = [
    "G",
    "G_prime",
    "G_prime_plus_0",
    "LocalModel",
    "fit_dkrgd",
    "fit_krgd",
    "gaussian_kernel",
    "influence_weight",
    "kernel_matrix",
    "l_sigma",
    "partition",
    "predict_dkrgd",
    "predict_krgd",
    "step_size_bound",
]
