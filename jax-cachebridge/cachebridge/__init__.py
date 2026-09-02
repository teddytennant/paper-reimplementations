"""CacheBridge: closed-form cross-model KV cache transfer (arXiv:2609.00891)."""

from .fused_fit import fused_fit, streaming_stats
from .pipeline import Mapper, apply_mapper, attn_repair_weights, fit_mapper
from .ridge import apply_affine, fit_affine, solve_from_stats, sufficient_statistics

__all__ = [
    "Mapper",
    "apply_affine",
    "apply_mapper",
    "attn_repair_weights",
    "fit_affine",
    "fit_mapper",
    "fused_fit",
    "solve_from_stats",
    "streaming_stats",
    "sufficient_statistics",
]
