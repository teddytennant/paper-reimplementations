"""Simulated DTQW quantum diffusion kernel (arXiv:2609.31070)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from qd_dtqw.losses import (
    categorical_cross_entropy,
    histogram_kl,
    intensity_histogram,
    mixed_loss,
    time_alpha,
)
from qd_dtqw.noise import amplitude_damping, depolarize
from qd_dtqw.reverse import (
    generate,
    init_reverse,
    reverse_logits,
    reverse_step,
)
from qd_dtqw.walk import (
    DTQW,
    basis_state,
    one_step_unitary,
    position_probs,
    position_projectors,
)

__all__ = [
    "DTQW",
    "amplitude_damping",
    "basis_state",
    "categorical_cross_entropy",
    "depolarize",
    "generate",
    "histogram_kl",
    "init_reverse",
    "intensity_histogram",
    "mixed_loss",
    "one_step_unitary",
    "position_probs",
    "position_projectors",
    "reverse_logits",
    "reverse_step",
    "time_alpha",
]
