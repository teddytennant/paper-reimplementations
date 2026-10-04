"""Approximately shifted robust Halpern iteration (arXiv 2609.28792)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

from rash_halpern.halpern import algorithm1
from rash_halpern.models import MODELS, model_i, model_ii, model_iii
from rash_halpern.operator import bellman, certificate_residual, greedy_policy

__all__ = [
    "MODELS",
    "algorithm1",
    "bellman",
    "certificate_residual",
    "greedy_policy",
    "model_i",
    "model_ii",
    "model_iii",
]
