"""Directed node-arc incidence and conservation helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def node_arc_incidence(tails, heads, n_nodes: int):
    """Directed node-arc incidence: C[i,e] = +1 if e leaves i, -1 if e enters i."""
    tails = np.asarray(tails, dtype=int)
    heads = np.asarray(heads, dtype=int)
    m = tails.shape[0]
    C = np.zeros((n_nodes, m), dtype=np.float64)
    for e in range(m):
        C[tails[e], e] = 1.0
        C[heads[e], e] = -1.0
    return jnp.asarray(C, dtype=jnp.float64)


def conservation_residual(C, f, d):
    """C f - d."""
    C = jnp.asarray(C, dtype=jnp.float64)
    f = jnp.asarray(f, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    return C @ f - d
