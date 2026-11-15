"""MoARa: module-aware rank allocation and structure-preserving decomposition."""

from moara.align import alignment_score, cosine_alignment, module_alignment
from moara.decompose import decompose, recompose, repeat_blocks
from moara.optim import MoARaState, init_moara_state, moara_update
from moara.project import project, reconstruct_direction, truncated_svd
from moara.rank import (
    R_BASE_1B,
    R_BASE_7B,
    R_BASE_350M,
    allocate_ranks,
    default_block_size,
    default_r_base,
)

__all__ = [
    "MoARaState",
    "R_BASE_1B",
    "R_BASE_7B",
    "R_BASE_350M",
    "alignment_score",
    "allocate_ranks",
    "cosine_alignment",
    "decompose",
    "default_block_size",
    "default_r_base",
    "init_moara_state",
    "moara_update",
    "module_alignment",
    "project",
    "recompose",
    "reconstruct_direction",
    "repeat_blocks",
    "truncated_svd",
]
