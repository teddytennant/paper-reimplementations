"""Hellinger distance between discrete distributions.

The paper writes H for the Hellinger distance and defines the population
contextual dissimilarity in Section 2.3 as the squared distance

    Gamma_h(v1, v2) = H^2(p_{v1}(X_{j,c}|v1), p_{v2}(X_{j,c}|v2)).

We follow the paper's normalization, H^2(p, q) = 1 - sum_x sqrt(p_x q_x), so
that H^2 lands in [0, 1] and H itself is a metric.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def bhattacharyya_coefficient(p, q):
    """sum_x sqrt(p_x q_x), the overlap that the paper calls 1 - H^2."""
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    return jnp.sum(jnp.sqrt(p * q), axis=-1)


def squared_hellinger(p, q):
    """H^2(p, q) = 1 - sum_x sqrt(p_x q_x).

    Clipped at zero from below because the overlap can exceed one by a rounding
    error when p and q are the same array.
    """
    return jnp.maximum(1.0 - bhattacharyya_coefficient(p, q), 0.0)


def hellinger(p, q):
    """H(p, q), the square root of :func:`squared_hellinger`."""
    return jnp.sqrt(squared_hellinger(p, q))


def pairwise_squared_hellinger(P):
    """Matrix of H^2 between the rows of P.

    P has shape (n, support). Returns an (n, n) matrix with a zero diagonal.
    """
    P = jnp.asarray(P)
    root = jnp.sqrt(P)
    overlap = root @ root.T
    out = jnp.maximum(1.0 - overlap, 0.0)
    return out - jnp.diag(jnp.diag(out))


def pairwise_hellinger(P):
    """Matrix of H between the rows of P."""
    return jnp.sqrt(pairwise_squared_hellinger(P))
