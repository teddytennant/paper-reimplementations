"""Variational quantum transformer (arXiv:2609.18565) in JAX."""

from jax_vqt.data import (
    CONTEXT,
    DET_CYCLE,
    LEX_WORDS,
    VOCAB,
    grammar_score,
    make_dataset,
    windows,
)
from jax_vqt.model import (
    VQTConfig,
    batch_nll,
    count_params,
    encode_context,
    init_params,
    next_token_probs,
    nll_loss,
    parameter_count,
)
from jax_vqt.train import greedy_generate, make_step

__all__ = [
    "CONTEXT",
    "DET_CYCLE",
    "LEX_WORDS",
    "VOCAB",
    "VQTConfig",
    "batch_nll",
    "count_params",
    "encode_context",
    "grammar_score",
    "greedy_generate",
    "init_params",
    "make_dataset",
    "make_step",
    "next_token_probs",
    "nll_loss",
    "parameter_count",
    "windows",
]
