"""FCx: feasible counterfactual explanations in JAX (arXiv:2609.19383)."""

from jax_fcx.classifier import (
    forward as classifier_forward,
    init_classifier,
    train_classifier,
)
from jax_fcx.constraints import binary_penalty, feasibility_loss, unary_penalty
from jax_fcx.lof import lof_scores, mean_lof
from jax_fcx.losses import combine_losses, hinge_validity, kl_standard_normal
from jax_fcx.metrics import feasibility_rate, proximity, sparsity_score, validity_rate
from jax_fcx.train import FCxConfig, make_synthetic_data, train_fcx
from jax_fcx.vae import assemble_counterfactual, decode, encode, generate_cf, init_cvae

__version__ = "0.1.0"

__all__ = [
    "FCxConfig",
    "assemble_counterfactual",
    "binary_penalty",
    "classifier_forward",
    "combine_losses",
    "decode",
    "encode",
    "feasibility_loss",
    "feasibility_rate",
    "generate_cf",
    "hinge_validity",
    "init_classifier",
    "init_cvae",
    "kl_standard_normal",
    "lof_scores",
    "make_synthetic_data",
    "mean_lof",
    "proximity",
    "sparsity_score",
    "train_classifier",
    "train_fcx",
    "unary_penalty",
    "validity_rate",
]
