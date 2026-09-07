"""Section 6.5 of arXiv:2609.04420, the score map across retraining windows.

This part falls out of the sampling work almost for free, so it is here, but it is a
different argument: the map u is the empirical distribution function of a model's own
validation scores evaluated at a test score, and the claim is about what that map does
to within-fold rankings.
"""

import jax
import jax.numpy as jnp


def rank_map(validation_scores: jax.Array, scores: jax.Array) -> jax.Array:
    """u(q) = (1/n_V) sum_j I{q_j^V <= q}, Equation (22)'s map, nondecreasing in q."""
    v = jnp.sort(jnp.asarray(validation_scores))
    return jnp.searchsorted(v, jnp.asarray(scores), side="right") / v.shape[0]


def conformal_rank_map(validation_scores: jax.Array, scores: jax.Array) -> jax.Array:
    """utilde(q) = (1 + sum_j I{q_j^V <= q}) / (n_V + 1), Equation (19).

    Proposition 15: under exchangeability of the validation scores and a test score,
    Pr{utilde(Q) <= alpha} <= alpha, and under Assumption 2 with a purge satisfying (4)
    the bound degrades to alpha + 2 beta(g - w - h - Delta).
    """
    v = jnp.sort(jnp.asarray(validation_scores))
    counts = jnp.searchsorted(v, jnp.asarray(scores), side="right")
    return (1.0 + counts) / (v.shape[0] + 1.0)


def roc_auc(labels: jax.Array, scores: jax.Array) -> jax.Array:
    """Probability that a random positive outranks a random negative, ties at 1/2."""
    labels = jnp.asarray(labels)
    order = jnp.argsort(jnp.asarray(scores))
    y = labels[order]
    ranks = _midranks(jnp.asarray(scores)[order])
    n1 = jnp.sum(y == 1)
    n0 = jnp.sum(y == 0)
    r1 = jnp.sum(jnp.where(y == 1, ranks, 0.0))
    return (r1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def average_precision(labels: jax.Array, scores: jax.Array) -> jax.Array:
    """Sum over positives of precision at that positive's rank, divided by N_1.

    Ties are broken by index, which is why Proposition 13 needs u to be injective on the
    fold's scores for the equality to be exact.
    """
    labels = jnp.asarray(labels)
    order = jnp.argsort(-jnp.asarray(scores), stable=True)
    y = labels[order].astype(jnp.float64)
    cum = jnp.cumsum(y)
    precision = cum / jnp.arange(1, y.shape[0] + 1, dtype=jnp.float64)
    return jnp.sum(precision * y) / jnp.sum(y)


def _midranks(sorted_scores: jax.Array) -> jax.Array:
    """1-based ranks, averaged inside each tied group."""
    lo = jnp.searchsorted(sorted_scores, sorted_scores, side="left").astype(jnp.float64)
    hi = jnp.searchsorted(sorted_scores, sorted_scores, side="right").astype(jnp.float64)
    return (lo + hi + 1.0) / 2.0
