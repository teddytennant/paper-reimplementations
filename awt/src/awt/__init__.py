"""Activation-aware weight tensorization (arXiv 2610.10085)."""

import jax

# TT-SVD on the default float32 dtype can miss a 1e-4 reconstruction check
# once K grows past a 4x4. Enable float64 before any arrays are built.
jax.config.update("jax_enable_x64", True)

from awt.compress import compress, deploy
from awt.importance import column_scales, importance_mean_abs, importance_std
from awt.tt import densify, tt_matvec, tt_svd

__all__ = [
    "column_scales",
    "compress",
    "densify",
    "deploy",
    "importance_mean_abs",
    "importance_std",
    "tt_matvec",
    "tt_svd",
]

__version__ = "0.1.0"
