"""Suan preference optimization (arXiv:2609.08634)."""

from suan.losses import (
    DEFAULT_BETA,
    DEFAULT_TAU,
    preference_gradient_coefficient,
    preference_loss,
    regularizer_gradient_coefficient,
    regularizer_loss,
    suan_loss,
)
from suan.policy import log_prob, log_softmax_rows

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_TAU",
    "log_prob",
    "log_softmax_rows",
    "preference_gradient_coefficient",
    "preference_loss",
    "regularizer_gradient_coefficient",
    "regularizer_loss",
    "suan_loss",
]
