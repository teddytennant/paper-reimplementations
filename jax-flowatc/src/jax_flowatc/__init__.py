"""JAX implementation of FlowATC (arXiv:2609.16528)."""

from jax_flowatc.baselines import constant_velocity, constant_velocity_ade
from jax_flowatc.cfm import (
    cfm_loss_batch,
    cfm_loss_one,
    euler_integrate,
    euler_path,
    interpolant,
    sample_completions,
    sample_logit_normal,
    target_velocity,
)
from jax_flowatc.config import FlowATCConfig, paper_tiny_config, tiny_config
from jax_flowatc.data import (
    ZScoreStats,
    apply_zscore,
    deltas_and_elapsed,
    invert_zscore,
    synthetic_straight,
    synthetic_with_turn,
    zscore_stats,
)
from jax_flowatc.layers import adaln, adaln_from_stats, block_causal_mask
from jax_flowatc.metrics import ade, fde, min_ade_at_k, min_fde_at_k
from jax_flowatc.model import (
    FlowATCParams,
    apply_one_block,
    block_attention_weights,
    eq11_concat,
    hidden_states,
    init_params,
    velocity_field,
)
from jax_flowatc.train import AdamWState, adamw_init, make_train_step, train_step

__all__ = [
    "AdamWState",
    "FlowATCConfig",
    "FlowATCParams",
    "ZScoreStats",
    "adaln",
    "adaln_from_stats",
    "adamw_init",
    "ade",
    "apply_one_block",
    "apply_zscore",
    "block_attention_weights",
    "block_causal_mask",
    "cfm_loss_batch",
    "cfm_loss_one",
    "constant_velocity",
    "constant_velocity_ade",
    "deltas_and_elapsed",
    "eq11_concat",
    "euler_integrate",
    "euler_path",
    "fde",
    "hidden_states",
    "init_params",
    "interpolant",
    "invert_zscore",
    "make_train_step",
    "min_ade_at_k",
    "min_fde_at_k",
    "paper_tiny_config",
    "sample_completions",
    "sample_logit_normal",
    "synthetic_straight",
    "synthetic_with_turn",
    "target_velocity",
    "tiny_config",
    "train_step",
    "velocity_field",
    "zscore_stats",
]
