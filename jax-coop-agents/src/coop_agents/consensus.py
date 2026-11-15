"""Table I cooperation strategies: voter, majority, MM-LOO, weighted average, simplicial, SM-LOO."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from coop_agents.losses import confidence


def _n_classes(probs: jax.Array) -> int:
    return probs.shape[-1]


def _one_hot_votes(probs: jax.Array) -> jax.Array:
    """One-hot argmax ballots. Ties take the smallest index (JAX argmax)."""
    idx = jnp.argmax(probs, axis=-1)
    return jax.nn.one_hot(idx, _n_classes(probs), dtype=probs.dtype)


def voter(probs: jax.Array, key: jax.Array) -> jax.Array:
    """Voter model: pick an agent uniformly, return its softmax vector y^{k*}."""
    k = probs.shape[0]
    idx = jax.random.randint(key, (), minval=0, maxval=k)
    return probs[idx]


def majority(probs: jax.Array) -> jax.Array:
    """Majority of argmax votes as a one-hot vector (smallest-index tie-break).

    Paper eq. (7) typesets a sum of softmax vectors; this follows the majority
    of {argmax y^k} used in Table I and the tests.
    """
    votes = _one_hot_votes(probs)
    counts = jnp.sum(votes, axis=0)
    idx = jnp.argmax(counts, axis=-1)
    return jax.nn.one_hot(idx, _n_classes(probs), dtype=probs.dtype)


def majority_loo(probs: jax.Array) -> jax.Array:
    """Leave-one-out majority: for each agent k, majority of the others.

    Returns consensus with a leading agent axis, shape (K, C) or (K, N, C).
    Ties break toward the smallest class index.
    """
    votes = _one_hot_votes(probs)
    total = jnp.sum(votes, axis=0, keepdims=True)
    loo_counts = total - votes
    idx = jnp.argmax(loo_counts, axis=-1)
    return jax.nn.one_hot(idx, _n_classes(probs), dtype=probs.dtype)


def weighted_average(probs: jax.Array, weights: jax.Array) -> jax.Array:
    """ỹ_c = sum_k w_k y_c^k / sum_k w_k. `probs` leading axis is agents."""
    w = jnp.asarray(weights, dtype=probs.dtype)
    while w.ndim < probs.ndim:
        w = jnp.expand_dims(w, axis=-1)
    numer = jnp.sum(w * probs, axis=0)
    denom = jnp.sum(w, axis=0)
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    return numer / denom


def simplicial(
    probs: jax.Array,
    weights: jax.Array | None = None,
    loo: bool = False,
) -> jax.Array:
    """All-to-all confidence-weighted consensus on the simplex (paper eq. 10).

    Not a graph 1-simplex / pairwise KL. If `loo`, each agent sees the
    weighted average of the others (SM-LOO).
    """
    if weights is None:
        weights = confidence(probs)
    w = jnp.asarray(weights, dtype=probs.dtype)
    if not loo:
        return weighted_average(probs, w)
    # Broadcast weights to (K, ...) matching probs without the class axis.
    w_lead = w
    while w_lead.ndim < probs.ndim - 1:
        w_lead = jnp.expand_dims(w_lead, axis=-1)
    # w_lead: (K,) or (K, N); expand for class multiply
    w_full = jnp.expand_dims(w_lead, axis=-1)
    total_wy = jnp.sum(w_full * probs, axis=0, keepdims=True)
    total_w = jnp.sum(w_full, axis=0, keepdims=True)
    loo_wy = total_wy - w_full * probs
    loo_w = total_w - w_full
    loo_w = jnp.where(loo_w > 0, loo_w, jnp.ones_like(loo_w))
    return loo_wy / loo_w
