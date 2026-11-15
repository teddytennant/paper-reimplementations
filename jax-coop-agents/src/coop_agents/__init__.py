"""Multi-agent cooperation-driven training (arXiv:2609.16917)."""

from coop_agents.consensus import (
    majority,
    majority_loo,
    simplicial,
    voter,
    weighted_average,
)
from coop_agents.losses import (
    combined_loss,
    confidence,
    cross_entropy,
    entropy,
    group_loss,
    isolated_loss,
    softmax_weights,
)
from coop_agents.models import apply_mlp, count_params, mlp, softmax
from coop_agents.train import init_opt_state, step

__all__ = [
    "apply_mlp",
    "combined_loss",
    "confidence",
    "count_params",
    "cross_entropy",
    "entropy",
    "group_loss",
    "init_opt_state",
    "isolated_loss",
    "majority",
    "majority_loo",
    "mlp",
    "simplicial",
    "softmax",
    "softmax_weights",
    "step",
    "voter",
    "weighted_average",
]
