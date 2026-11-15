"""Operator-aligned Muon for convolutional kernels (arXiv:2609.09676)."""

from muon_c.geometry import (
    conv_norm_grid,
    critical_fft,
    critical_ifft,
    lebesgue_constant,
    patch_norm,
    rms,
    sampled_norm,
    spatial_norm,
    spatial_polar_direction,
    transfer_matrix,
    unfold,
)
from muon_c.optim import (
    muon_c_direction,
    muon_c_scale,
    muon_c_step,
    nesterov_lookahead,
    unfolded_muon_direction,
    unfolded_scale,
    uses_muon_c,
)
from muon_c.polar import polar_ns, polar_svd

__all__ = [
    "conv_norm_grid",
    "critical_fft",
    "critical_ifft",
    "lebesgue_constant",
    "muon_c_direction",
    "muon_c_scale",
    "muon_c_step",
    "nesterov_lookahead",
    "patch_norm",
    "polar_ns",
    "polar_svd",
    "rms",
    "sampled_norm",
    "spatial_norm",
    "spatial_polar_direction",
    "transfer_matrix",
    "unfold",
    "unfolded_muon_direction",
    "unfolded_scale",
    "uses_muon_c",
]
