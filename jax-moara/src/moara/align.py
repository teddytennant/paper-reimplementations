"""Alignment diagnostic (eq. 2)."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

Array = jnp.ndarray


def cosine_alignment(G: Array, G_recon: Array) -> Array:
    """S = cos(G, G_recon) = <G, G_recon> / (||G||_2 ||G_recon||_2)."""
    g = jnp.ravel(jnp.asarray(G))
    h = jnp.ravel(jnp.asarray(G_recon))
    num = jnp.vdot(g.astype(jnp.float64), h.astype(jnp.float64)).astype(g.dtype)
    den = jnp.linalg.norm(g) * jnp.linalg.norm(h)
    return num / jnp.maximum(den, jnp.asarray(0, dtype=den.dtype))


def alignment_score(G: Array, G_recon: Array) -> Array:
    """Alias of ``cosine_alignment`` (eq. 2)."""
    return cosine_alignment(G, G_recon)


def module_alignment(scores_by_type: Mapping[str, Array]) -> dict[str, Array]:
    """S_τ = mean over profiling scores of matrices of type τ."""
    out = {}
    for tau, scores in scores_by_type.items():
        arr = jnp.asarray(scores)
        out[tau] = jnp.mean(arr)
    return out
