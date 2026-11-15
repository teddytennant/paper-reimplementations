"""Direct Spectral Laplacian Estimation from effective resistance."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from dsle.estimate import (
    dsle,
    laplacian_from_covariance,
    laplacian_from_resistance,
    maximum_spanning_tree,
    mds_plus,
    regularized_resistance,
    resistance_from_covariance,
    sample_covariance,
    sparsify,
)

__all__ = [
    "dsle",
    "laplacian_from_covariance",
    "laplacian_from_resistance",
    "maximum_spanning_tree",
    "mds_plus",
    "regularized_resistance",
    "resistance_from_covariance",
    "sample_covariance",
    "sparsify",
]

__version__ = "0.1.0"
