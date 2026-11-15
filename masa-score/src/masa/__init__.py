"""Matrix Approximation Sparse Attention scores."""

from masa._host import configure_host

configure_host()

from masa.score import (
    block_score,
    clip_scores,
    pair_score,
    rank_units,
    reconstruction_residual,
    slash_offsets,
    slash_score,
    soft_balance,
    vertical_score,
)

__all__ = [
    "block_score",
    "clip_scores",
    "pair_score",
    "rank_units",
    "reconstruction_residual",
    "slash_offsets",
    "slash_score",
    "soft_balance",
    "vertical_score",
]
