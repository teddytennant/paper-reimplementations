"""Effective depth from linear CKA of layer activations."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from effective_depth.diagnostic import (
    cka,
    deff,
    deff_from_rho,
    deff_on_updates,
    drop_top_pcs,
    first_non_bos,
    mean_pool,
    orthogonal_update_population_rho,
    token_normed,
)

__all__ = [
    "cka",
    "deff",
    "deff_from_rho",
    "deff_on_updates",
    "drop_top_pcs",
    "first_non_bos",
    "mean_pool",
    "orthogonal_update_population_rho",
    "token_normed",
]
