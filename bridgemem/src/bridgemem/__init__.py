"""BridgeMem: causal dyadic transition residuals (arXiv:2609.29268)."""

from bridgemem.constants import EPS, K, TAU, WINDOW
from bridgemem.kernel import (
    A_ct,
    A_nn,
    attention_pool,
    event_embedding,
    final_log_score,
    g_gate,
    init_params,
    log_scores,
    mix_weight_count,
    mixed_residual,
    recency_bin,
    residual_from_tokens,
    rho_B,
)
from bridgemem.memory import fit_count_table, retrieve_pair, stack_candidates

__all__ = [
    "A_ct",
    "A_nn",
    "EPS",
    "K",
    "TAU",
    "WINDOW",
    "attention_pool",
    "event_embedding",
    "final_log_score",
    "fit_count_table",
    "g_gate",
    "init_params",
    "log_scores",
    "mix_weight_count",
    "mixed_residual",
    "recency_bin",
    "residual_from_tokens",
    "retrieve_pair",
    "rho_B",
    "stack_candidates",
]
