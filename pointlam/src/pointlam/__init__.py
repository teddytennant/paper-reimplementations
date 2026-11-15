"""PointLAM kernels (arXiv:2609.21780): LPS, LHA, and a tiny LAM block."""

from pointlam.bdm import BDMParams, apply_bdm, init_bdm
from pointlam.grid import cell_indices, neighbor_mask
from pointlam.lam import (
    LAMBlockParams,
    PointLAMParams,
    apply_lam_block,
    apply_pointlam,
    init_lam_block,
    init_pointlam,
)
from pointlam.lha import (
    LHAParams,
    apply_lha,
    init_lha,
    layer_norm,
    local_hadamard,
    residual_hadamard,
)
from pointlam.lps import apply_lps, deviation, dss_indices, init_mlp, mlp_apply
from pointlam.nn import Linear, apply_linear, init_linear

__all__ = [
    "BDMParams",
    "LAMBlockParams",
    "LHAParams",
    "Linear",
    "PointLAMParams",
    "apply_bdm",
    "apply_lam_block",
    "apply_lha",
    "apply_linear",
    "apply_lps",
    "apply_pointlam",
    "cell_indices",
    "deviation",
    "dss_indices",
    "init_bdm",
    "init_lam_block",
    "init_lha",
    "init_linear",
    "init_mlp",
    "init_pointlam",
    "layer_norm",
    "local_hadamard",
    "mlp_apply",
    "neighbor_mask",
    "residual_hadamard",
]
