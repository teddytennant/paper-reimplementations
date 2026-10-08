"""Sink-isolated Rényi pooling, equations (3) and (5).

Attention arrives as (G, r, Q, P): groups, heads per group, query
observers, positions. Heads with sink mass below z_min are ineligible.
"""

import jax.numpy as jnp


def sink_mass(attention, sink):
    """z_h = sum_{j in S} a_j, per head. attention is (G, r, Q, P)."""
    return jnp.sum(attention * sink.reshape((1, 1, 1, -1)), axis=-1)


def collision(residual):
    """C_h = sum_j e_{h,j}^2 on the non-sink positions, equation (3).

    residual is (G, r, Q, P). The sum is over positions, not queries.
    """
    return jnp.sum(residual**2, axis=-1)


def renyi2(collision_value):
    """H_2 = -ln C, equation (3)."""
    return -jnp.log(jnp.maximum(collision_value, 1e-30))


def pool(attention, weights):
    """Pooled distribution mu, equation (5).

    weights has shape (G, r) and sums to one over every group. The pool
    averages groups and query observers with equal weight.
    """
    g, _, q, _ = attention.shape
    weighted = attention * weights[:, :, None, None]
    return jnp.sum(weighted, axis=(0, 1, 2)) / (g * q)
