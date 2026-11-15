"""JAX kernel of Decision-Relevant Fresh Comparison (arXiv:2609.16824)."""

from jax_drfc.certificates import (
    PAPER_C,
    SEQ_A0,
    SEQ_C0,
    anytime_radius,
    beta_radius,
    eliminate_once,
    equal_agent_means,
    lower_confidence_gap,
    pairwise_ghat_and_B,
)
from jax_drfc.drfc import DRFCConfig, DRFCResult, run_drfc, run_one_scan
from jax_drfc.drfc_seq import DRFCSeqConfig, DRFCSeqResult, run_drfc_seq, seq_switch_decision
from jax_drfc.env import (
    HeterogeneousBandit,
    aggregate_block,
    balanced_block_schedule,
    cancellation_mean_path,
    count_s_loc_s_dec,
    make_cancellation_env,
)

__all__ = [
    "DRFCConfig",
    "DRFCResult",
    "DRFCSeqConfig",
    "DRFCSeqResult",
    "HeterogeneousBandit",
    "PAPER_C",
    "SEQ_A0",
    "SEQ_C0",
    "aggregate_block",
    "anytime_radius",
    "balanced_block_schedule",
    "beta_radius",
    "cancellation_mean_path",
    "count_s_loc_s_dec",
    "eliminate_once",
    "equal_agent_means",
    "lower_confidence_gap",
    "make_cancellation_env",
    "pairwise_ghat_and_B",
    "run_drfc",
    "run_drfc_seq",
    "run_one_scan",
    "seq_switch_decision",
]
