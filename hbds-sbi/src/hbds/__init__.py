"""Checkable core of compositional diffusion SBI (arXiv 2609.36950)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from hbds.diffusion import beta_schedule, g_n_squared, reverse_sde_step
from hbds.gaussian import analytic_posterior, compositional_score
from hbds.hbds import hbds_sample, score_phi
from hbds.path import path_divergence, path_regularized_loss

__all__ = [
    "analytic_posterior",
    "beta_schedule",
    "compositional_score",
    "g_n_squared",
    "hbds_sample",
    "path_divergence",
    "path_regularized_loss",
    "reverse_sde_step",
    "score_phi",
]
