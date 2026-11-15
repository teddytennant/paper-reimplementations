"""ARACD kernel (arXiv:2609.21214): NCDM + ability residual + guess/slip affect."""

from aracd.affect import affect_mlp, guess_slip, modulate
from aracd.loss import binary_cross_entropy, supervised_loss
from aracd.model import ARACDConfig, forward, init_params
from aracd.ncdm import ncdm_base, ncdm_interaction, ncdm_mlp
from aracd.ops import clip_prob, layer_norm, logit, sigmoid
from aracd.residual import (
    GAMMA_SI_DEFAULT,
    ability_correct,
    ability_residual,
    attention_query,
    attention_scores,
    concept_residual,
    q_masked_softmax,
    stu_item_residual,
)

__version__ = "0.1.0"

__all__ = [
    "ARACDConfig",
    "GAMMA_SI_DEFAULT",
    "ability_correct",
    "ability_residual",
    "affect_mlp",
    "attention_query",
    "attention_scores",
    "binary_cross_entropy",
    "clip_prob",
    "concept_residual",
    "forward",
    "guess_slip",
    "init_params",
    "layer_norm",
    "logit",
    "modulate",
    "ncdm_base",
    "ncdm_interaction",
    "ncdm_mlp",
    "q_masked_softmax",
    "sigmoid",
    "stu_item_residual",
    "supervised_loss",
]
