"""Tiny categorical softmax policy over discrete prompts and actions."""

from __future__ import annotations

import jax


def log_softmax_rows(logits: jax.Array) -> jax.Array:
    """Row-wise log π_θ(a | x) for logits of shape (n_prompts, n_actions)."""
    return jax.nn.log_softmax(logits, axis=-1)


def softmax_rows(logits: jax.Array) -> jax.Array:
    """Row-wise π_θ(a | x)."""
    return jax.nn.softmax(logits, axis=-1)


def log_prob(
    logits: jax.Array,
    prompt_ids: jax.Array,
    action_ids: jax.Array,
) -> jax.Array:
    """log π_θ(y | x) for integer prompt and action ids."""
    return log_softmax_rows(logits)[prompt_ids, action_ids]
