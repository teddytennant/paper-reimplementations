"""hLLM: single pass decoding for generative reranking (arXiv:2609.01807).

The scoring head reads an N x K item-position score matrix off per-item hidden
states, and the Hungarian algorithm decodes all N ordinals from it at once.
Training relaxes that decode to a Sinkhorn plan so it has a gradient.
"""

from hllm.data import Slates, permutation_matrices, sample_slates
from hllm.heads import HeadConfig, apply_head, count_params, init_head
from hllm.hungarian import (
    assign_positions,
    assignment_score,
    decode_ranking,
    solve_assignment,
)
from hllm.metrics import exact_match, ndcg, top1_accuracy
from hllm.sinkhorn import (
    log_sinkhorn,
    permutation_cross_entropy,
    sinkhorn,
    sinkhorn_loss,
)

__all__ = [
    "HeadConfig",
    "Slates",
    "apply_head",
    "assign_positions",
    "assignment_score",
    "count_params",
    "decode_ranking",
    "exact_match",
    "init_head",
    "log_sinkhorn",
    "ndcg",
    "permutation_cross_entropy",
    "permutation_matrices",
    "sample_slates",
    "sinkhorn",
    "sinkhorn_loss",
    "solve_assignment",
    "top1_accuracy",
]
