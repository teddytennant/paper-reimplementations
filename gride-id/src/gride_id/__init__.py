"""Closed-form componentwise Gride intrinsic-dimension estimator."""

import jax

# Digamma differences and the alternating sum in I(gamma) need float64.
jax.config.update("jax_enable_x64", True)

from gride_id.estimator import (
    estimate_id,
    gride_kl,
    gride_mle,
    i_gamma,
    pairwise_ratios,
    von_mises_fit,
    von_mises_kl,
)

__all__ = [
    "estimate_id",
    "gride_kl",
    "gride_mle",
    "i_gamma",
    "pairwise_ratios",
    "von_mises_fit",
    "von_mises_kl",
]
