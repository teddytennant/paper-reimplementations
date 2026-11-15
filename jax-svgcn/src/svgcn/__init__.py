"""JAX implementation of SV-GCN (arXiv:2609.11680)."""

from svgcn.features import (
    EPS,
    angle_feature,
    angles_from_coords,
    motion_features,
    relative_motion,
)
from svgcn.layers import (
    downsample_mask,
    masked_gap,
    masked_group_norm,
    masked_group_norm_stats,
    make_mask,
)
from svgcn.model import (
    NUM_CLASSES,
    apply_svgcn,
    cross_entropy,
    init_svgcn,
)
from svgcn.skeleton import (
    ADJACENCY,
    ANGLE_TRIPLES,
    JOINT_NAMES,
    NUM_JOINTS,
    PARENTS,
    normalized_adjacency,
)

__all__ = [
    "ADJACENCY",
    "ANGLE_TRIPLES",
    "EPS",
    "JOINT_NAMES",
    "NUM_CLASSES",
    "NUM_JOINTS",
    "PARENTS",
    "angle_feature",
    "angles_from_coords",
    "apply_svgcn",
    "cross_entropy",
    "downsample_mask",
    "init_svgcn",
    "make_mask",
    "masked_gap",
    "masked_group_norm",
    "masked_group_norm_stats",
    "motion_features",
    "normalized_adjacency",
    "relative_motion",
]
