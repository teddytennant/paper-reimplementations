"""Differentiable bicausal optimal transport on enumerated Markov chains."""

import jax

jax.config.update("jax_enable_x64", True)

from dbot.core import (
    bisimulation_lp,
    envelope_grad,
    fit_kernel,
    occupancy,
    softmax_kernel,
)

__all__ = [
    "bisimulation_lp",
    "envelope_grad",
    "fit_kernel",
    "occupancy",
    "softmax_kernel",
]
