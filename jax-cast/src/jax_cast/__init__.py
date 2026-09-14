"""CAST: Canonical Approximate Schur Tree (arXiv:2609.09255)."""

from jax_cast.chol import (
    CastFactor,
    PivotStar,
    apply_factor,
    cast_chol,
    conductance_laplacian,
    factor_laplacian,
    factor_preconditioner,
    pcg,
)
from jax_cast.prufer import prufer_decode, prufer_encode
from jax_cast.schur import (
    cast_schur_tree,
    exact_schur_clique,
    inverse_marginal_weight,
    is_connected,
    laplacian_from_edges,
    leverage_scores,
)

__all__ = [
    "CastFactor",
    "PivotStar",
    "apply_factor",
    "cast_chol",
    "cast_schur_tree",
    "conductance_laplacian",
    "exact_schur_clique",
    "factor_laplacian",
    "factor_preconditioner",
    "inverse_marginal_weight",
    "is_connected",
    "laplacian_from_edges",
    "leverage_scores",
    "pcg",
    "prufer_decode",
    "prufer_encode",
]
