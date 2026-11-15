"""Objective and tracking-error helpers."""

from __future__ import annotations

import jax.numpy as jnp


def objective(problem, w):
    """Exact outer objective F(w) = (1/m) sum_i f_i(g_i(w))."""
    return problem.F(w)


def tracking_error(u, g):
    """||u - g(w)||_2 over the stacked inner blocks."""
    return jnp.linalg.norm(jnp.ravel(u - g))


def mean_square_tracking_error(u, g):
    return jnp.mean((u - g) ** 2)
