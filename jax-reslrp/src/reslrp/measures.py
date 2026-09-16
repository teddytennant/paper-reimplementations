"""Cancellation C and amplification A (Eq. 2)."""

from __future__ import annotations

import jax.numpy as jnp


def _sum_abs(x):
    return jnp.sum(jnp.abs(x))


def cancellation(z_in, z_up, eps: float = 0.0):
    """Forward cancellation C (Eq. 2).

    C = sum(|z_in| + |z_up|) / sum(|z_out|), with z_out = z_in + z_up.

    C = 1 means no cancellation (same-sign additions); C >> 1 means
    destructive interference. Reduction is a global ratio of sums over
    batch and channel axes, as in the paper.
    """
    z_in = jnp.asarray(z_in)
    z_up = jnp.asarray(z_up)
    z_out = z_in + z_up
    numer = _sum_abs(z_in) + _sum_abs(z_up)
    denom = _sum_abs(z_out) + jnp.asarray(eps, dtype=z_in.dtype)
    return numer / denom


def amplification(z_in, z_out, eps: float = 0.0):
    """Forward mass ratio A_fwd = sum(|z_in|) / sum(|z_out|).

    Large under cancellation; equals 1 when z_up = 0 (z_out = z_in).
    Paper Eq. 2 defines A on relevances; see ``relevance_amplification``.
    """
    z_in = jnp.asarray(z_in)
    z_out = jnp.asarray(z_out)
    denom = _sum_abs(z_out) + jnp.asarray(eps, dtype=z_in.dtype)
    return _sum_abs(z_in) / denom


def relevance_amplification(r_in, r_up, r_out, eps: float = 0.0):
    """Relevance amplification A (Eq. 2).

    A = sum(|R_in| + |R_up|) / sum(|R_out|). Under LRP-γ this is bounded
    by 1 + 2/γ (Eq. 5).
    """
    r_in = jnp.asarray(r_in)
    r_up = jnp.asarray(r_up)
    r_out = jnp.asarray(r_out)
    numer = _sum_abs(r_in) + _sum_abs(r_up)
    denom = _sum_abs(r_out) + jnp.asarray(eps, dtype=r_in.dtype)
    return numer / denom
