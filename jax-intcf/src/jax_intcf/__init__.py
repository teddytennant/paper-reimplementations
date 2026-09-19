"""Interpretable causal forests (IntCF), arXiv:2609.16971."""

from jax_intcf.criteria import (
    bias_criterion,
    bias_validation,
    difference_in_means,
    evaluate_split,
    het_criterion,
    het_validation,
    intcf_delta,
    lemma4_mse_decrease,
    mean_squared_change,
    signed_bias,
    signed_het,
    unsigned_from_signed,
)
from jax_intcf.forest import IntCF
from jax_intcf.tree import IntCT

__version__ = "0.1.0"

__all__ = [
    "IntCF",
    "IntCT",
    "bias_criterion",
    "bias_validation",
    "difference_in_means",
    "evaluate_split",
    "het_criterion",
    "het_validation",
    "intcf_delta",
    "lemma4_mse_decrease",
    "mean_squared_change",
    "signed_bias",
    "signed_het",
    "unsigned_from_signed",
]
