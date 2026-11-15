"""A synthetic stand-in for the offline teacher distillation set.

Phase 1 of Section 4.3 runs a teacher ranker over every training slate once and
stores the permutation it decodes. Phase 2 never calls the teacher again; the
student only ever sees fixed (hidden states, permutation matrix) pairs. That is
the whole interface between the two phases, so a synthetic generator that emits
the same pairs exercises the real training path.

Here a slate is N item vectors drawn from a standard normal, standing in for
the backbone's readout hidden states, and the teacher's relevance is a fixed
linear functional of them plus noise. The teacher permutation sorts by that
relevance, descending. It is a much easier signal than a real teacher's, and it
is the same shape.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["Slates", "sample_slates", "teacher_direction", "permutation_matrices"]


class Slates(NamedTuple):
    """One batch of slates.

    hidden:   (B, N, D) per-item vectors, the head's input.
    order:    (B, N) item index at each rank position, the teacher's ranking.
    teacher:  (B, N, N) permutation matrices P_teacher for equation (3).
    relevance:(B, N) the latent score the teacher sorted by.
    """

    hidden: jnp.ndarray
    order: jnp.ndarray
    teacher: jnp.ndarray
    relevance: jnp.ndarray


def teacher_direction(d_model: int, seed: int = 0):
    """The fixed direction the synthetic teacher scores items along."""
    w = jax.random.normal(jax.random.key(seed), (d_model,))
    return w / jnp.linalg.norm(w)


def permutation_matrices(order, n_positions: int):
    """Turn rankings into P_teacher.

    ``order[..., j]`` is the item at rank j, so ``P[i, j] = 1`` exactly when
    item i was ranked at position j.
    """
    order = jnp.asarray(order)
    n_items = order.shape[-1]
    rank = jnp.argsort(order, axis=-1)  # rank[i] = position of item i
    return jax.nn.one_hot(rank, n_positions, dtype=jnp.float32).reshape(
        order.shape[:-1] + (n_items, n_positions)
    )


def sample_slates(key, batch: int, n_items: int, d_model: int, noise: float = 0.1,
                  w=None, seed: int = 0) -> Slates:
    """Draw a batch of slates with their teacher permutations."""
    if w is None:
        w = teacher_direction(d_model, seed=seed)
    k_h, k_n = jax.random.split(key)
    hidden = jax.random.normal(k_h, (batch, n_items, d_model))
    relevance = hidden @ w + noise * jax.random.normal(k_n, (batch, n_items))
    order = jnp.argsort(-relevance, axis=-1)
    return Slates(
        hidden=hidden,
        order=order,
        teacher=permutation_matrices(order, n_items),
        relevance=relevance,
    )
