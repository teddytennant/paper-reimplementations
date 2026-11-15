"""Attention-head influence score, arXiv:2609.05074, in JAX."""

from .ablation import (
    accuracy,
    accuracy_for_masks,
    all_subset_masks,
    direct_path_logit_diff,
    head_mask,
    logit_diff,
)
from .influence import (
    Influence,
    alti_contributions,
    batched_influence,
    deltas,
    head_contributions,
    influence,
    influence_from_trace,
    influence_scores,
    layer_contributions,
    layer_scores,
    layer_weights,
    mean_scores_by_category,
    outcome_categories,
    proximity,
    s_final,
    signal_split,
    top_heads,
)
from .model import ModelConfig, Trace, forward, init_params, logit_direction
from .train import train

__all__ = [
    "ModelConfig", "Trace", "forward", "init_params", "logit_direction", "train",
    "Influence", "influence", "influence_from_trace", "batched_influence",
    "proximity", "alti_contributions", "head_contributions", "layer_contributions",
    "layer_weights", "deltas", "influence_scores", "s_final", "layer_scores",
    "signal_split", "top_heads", "outcome_categories", "mean_scores_by_category",
    "head_mask", "logit_diff", "direct_path_logit_diff", "accuracy",
    "accuracy_for_masks", "all_subset_masks",
]
