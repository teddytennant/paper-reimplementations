"""Decode-time calibration and N:M bitmask SpMV (arXiv 2610.12327)."""

from sparse_decoding.method import (
    apply_obs_removal,
    bits_per_nonzero,
    bitmask_spmv,
    build_x_ar,
    calibration_discrepancy,
    gram_hessian,
    nm_prune,
    obs_saliency_and_update,
    pack_bitmask,
    pack_values,
    population_hessian,
    projected_reconstruction_loss,
    reconstruction_loss,
)

__all__ = [
    "apply_obs_removal",
    "bits_per_nonzero",
    "bitmask_spmv",
    "build_x_ar",
    "calibration_discrepancy",
    "gram_hessian",
    "nm_prune",
    "obs_saliency_and_update",
    "pack_bitmask",
    "pack_values",
    "population_hessian",
    "projected_reconstruction_loss",
    "reconstruction_loss",
]
