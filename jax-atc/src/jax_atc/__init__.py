"""Abstract Token Curriculum (arXiv:2609.19717) in JAX."""

from jax_atc.data import parity_target, sample_parity, sample_subset
from jax_atc.model import (
    ATCConfig,
    ForwardResult,
    assemble_embeddings,
    build_cots,
    causal_attention,
    empty_cots,
    forward,
    init_params,
    mse_loss,
    sign_accuracy,
)
from jax_atc.train import (
    adamw_init,
    adamw_update,
    evaluate,
    make_step,
    run_curriculum,
    train_phase,
)

__all__ = [
    "ATCConfig",
    "ForwardResult",
    "adamw_init",
    "adamw_update",
    "assemble_embeddings",
    "build_cots",
    "causal_attention",
    "empty_cots",
    "evaluate",
    "forward",
    "init_params",
    "make_step",
    "mse_loss",
    "parity_target",
    "run_curriculum",
    "sample_parity",
    "sample_subset",
    "sign_accuracy",
    "train_phase",
]
