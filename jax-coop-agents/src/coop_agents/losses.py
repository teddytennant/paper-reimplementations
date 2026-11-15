"""Isolated, group, and combined losses from arXiv:2609.16917.

Isolated CE is paper eq. (4). Combined loss is the multiplicative form
eq. (3): L_k = L_k^{isol} * (L_k^{isol} + λ L_k^G).

`confidence` is 1 - H / log(C). Paper eq. (5) is `softmax_weights`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_EPS = 1e-12


def _as_probs(logits_or_probs: jax.Array) -> jax.Array:
    """Treat a simplex row as probabilities, otherwise softmax logits."""
    x = jnp.asarray(logits_or_probs)
    nonneg = jnp.all(x >= -1e-6, axis=-1, keepdims=True)
    sums_one = jnp.isclose(jnp.sum(x, axis=-1, keepdims=True), 1.0, atol=1e-4)
    use_probs = nonneg & sums_one
    probs = jnp.where(use_probs, jnp.clip(x, 0.0, 1.0), jax.nn.softmax(x, axis=-1))
    return probs


def _as_one_hot(labels: jax.Array, n_classes: int) -> jax.Array:
    labels = jnp.asarray(labels)
    if labels.shape[-1] == n_classes and jnp.issubdtype(labels.dtype, jnp.floating):
        return labels
    return jax.nn.one_hot(labels.astype(jnp.int32), n_classes)


def cross_entropy(logits_or_probs: jax.Array, labels: jax.Array) -> jax.Array:
    """Categorical CE along the last axis: -sum_c y_c log p_c.

    `logits_or_probs` may be logits or a probability simplex. `labels` may be
    one-hot or integer class indices.
    """
    probs = _as_probs(logits_or_probs)
    n_classes = probs.shape[-1]
    y = _as_one_hot(labels, n_classes)
    logp = jnp.log(jnp.clip(probs, _EPS, 1.0))
    return -jnp.sum(y * logp, axis=-1)


def entropy(probs: jax.Array) -> jax.Array:
    """Shannon entropy H(p) = -sum_c p_c log p_c (natural log)."""
    p = _as_probs(probs)
    logp = jnp.log(jnp.clip(p, _EPS, 1.0))
    return -jnp.sum(p * logp, axis=-1)


def confidence(probs: jax.Array) -> jax.Array:
    """Per-agent confidence in [0, 1]: w = 1 - H(p) / log(C).

    One-hot predictions give 1; uniform predictions give 0. Paper eq. (5)
    instead uses temperature-softmax of -H; see `softmax_weights`.
    """
    p = _as_probs(probs)
    n_classes = p.shape[-1]
    h = entropy(p)
    return 1.0 - h / jnp.log(jnp.asarray(n_classes, dtype=p.dtype))


def softmax_weights(probs: jax.Array, tau: float = 1.0) -> jax.Array:
    """Paper eq. (5): w_k = softmax_k(-H(y^k) / τ), summing to 1 over agents.

    `probs` has a leading agent axis. `tau > 0` is the temperature.
    """
    h = entropy(probs)
    logits = -h / jnp.asarray(tau, dtype=h.dtype)
    return jax.nn.softmax(logits, axis=0)


def isolated_loss(logits_or_probs: jax.Array, labels: jax.Array) -> jax.Array:
    """Paper eq. (4): L_k^{isol} = -sum_c y_c log(y_c^k)."""
    return cross_entropy(logits_or_probs, labels)


def group_loss(agent_probs: jax.Array, consensus: jax.Array) -> jax.Array:
    """L_k^G = CE(ỹ, y^k) = -sum_c ỹ_c log(y_c^k)."""
    return cross_entropy(agent_probs, consensus)


def combined_loss(isol: jax.Array, group: jax.Array, lam: float | jax.Array) -> jax.Array:
    """Paper eq. (3): L = L^{isol} * (L^{isol} + λ L^G). λ = 0 yields (L^{isol})^2."""
    isol = jnp.asarray(isol)
    group = jnp.asarray(group)
    lam = jnp.asarray(lam, dtype=isol.dtype)
    return isol * (isol + lam * group)
