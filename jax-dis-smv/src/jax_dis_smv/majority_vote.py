"""Majority vote predictor and empirical 0-1 risk (Eqs 1-2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def vote_scores(voter_preds: jax.Array, rho: jax.Array, n_classes: int = 2) -> jax.Array:
    """Inner expectation of Eq 1: score_y(x) = sum_j rho_j I[h_j(x)=y].

    Args:
        voter_preds: integer labels of shape (n, m).
        rho: simplex weights of shape (m,).
        n_classes: number of labels.

    Returns:
        scores of shape (n, n_classes).
    """
    rho = jnp.asarray(rho)
    oh = jax.nn.one_hot(voter_preds, n_classes, dtype=rho.dtype)
    return jnp.einsum("m,nmc->nc", rho, oh)


def majority_vote_predict(
    voter_preds: jax.Array, rho: jax.Array, n_classes: int = 2
) -> jax.Array:
    """Deterministic weighted majority vote MVrho of Eq 1.

    Ties break toward the smallest class index (JAX argmax).
    """
    scores = vote_scores(voter_preds, rho, n_classes)
    return jnp.argmax(scores, axis=-1)


def empirical_risk_01(
    voter_preds: jax.Array,
    rho: jax.Array,
    y: jax.Array,
    n_classes: int = 2,
) -> jax.Array:
    """Empirical 0-1 risk of MVrho on a finite sample (Eq 2)."""
    pred = majority_vote_predict(voter_preds, rho, n_classes)
    return jnp.mean(pred != y).astype(jnp.asarray(rho).dtype)


def empirical_risk_sigmoid(
    voter_preds: jax.Array,
    rho: jax.Array,
    y: jax.Array,
    slope: float = 100.0,
    n_classes: int = 2,
) -> jax.Array:
    """Sigmoid surrogate of the 0-1 loss, slope c as in Section 6 (default 100).

    Margin is score of the true label minus the strongest competing score.
    """
    scores = vote_scores(voter_preds, rho, n_classes)
    y = y.astype(jnp.int32)
    true_score = jnp.take_along_axis(scores, y[:, None], axis=1)[:, 0]
    mask = jax.nn.one_hot(y, n_classes, dtype=scores.dtype)
    best_other = jnp.max(scores - mask * 1.0e9, axis=1)
    margin = true_score - best_other
    return jnp.mean(jax.nn.sigmoid(-slope * margin))
