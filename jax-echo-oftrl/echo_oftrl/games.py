"""Finite normal-form games and the full payoff vectors players observe.

Section 2.1. Player i has action set [m], mixed strategy x_i in the simplex,
and utility U_i in [0,1] given as a tensor of shape (m,)*N. The observed
feedback at a profile x is

    u_i(x_{-i}) = grad_{x_i} U_i(x) in [0,1]^m,

multiaffine in the opponents and independent of x_i. Section 3.1's shifted
vector is g_i = u_i - <u_i, x_i> 1, so <g_i, x_i> = 0 and ||g_i||_inf <= 1.

Every player is given the same number of actions m here. The algorithm does
not need that; it keeps the whole run in one (N, m) array.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def payoff_vectors(utilities: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """u_i = grad_{x_i} U_i(x) for every i, stacked into shape (N, m).

    `utilities` has shape (N,) + (m,)*N; `x` has shape (N, m).
    """
    n = x.shape[0]
    out = []
    for i in range(n):
        t = jnp.moveaxis(utilities[i], i, 0)
        others = [j for j in range(n) if j != i]
        for j in others:
            t = jnp.tensordot(t, x[j], axes=([1], [0]))
        out.append(t)
    return jnp.stack(out, axis=0)


def shifted_gradients(u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """g_i = u_i - <u_i, x_i> 1, the G_i of Section 3.1."""
    return u - jnp.sum(u * x, axis=-1, keepdims=True)


def matching_pennies() -> jnp.ndarray:
    """Two players, two actions, constant sum 1. Unique equilibrium is uniform."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    return jnp.asarray(np.stack([a, 1.0 - a]))


def biased_matching_pennies() -> jnp.ndarray:
    """Two players, two actions, constant sum 1, payoff matrix [[1, 0], [0, 1/2]].

    The unique equilibrium is (1/3, 2/3) for both players with value 1/3, so
    unlike plain matching pennies the uniform start is not already a fixed
    point and the dynamics have somewhere to go.
    """
    a = np.array([[1.0, 0.0], [0.0, 0.5]])
    return jnp.asarray(np.stack([a, 1.0 - a]))


def rock_paper_scissors() -> jnp.ndarray:
    """Two players, three actions, constant sum 1. Unique equilibrium is uniform."""
    a = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    return jnp.asarray(np.stack([a, 1.0 - a]))


def random_game(num_players: int, num_actions: int, seed: int) -> jnp.ndarray:
    """Payoffs drawn uniformly from [0,1], the general-sum case of Theorem 3.3."""
    rng = np.random.default_rng(seed)
    shape = (num_players,) + (num_actions,) * num_players
    return jnp.asarray(rng.random(shape))


def duality_gap(utilities: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Nash gap of a profile in a two-player constant-sum game.

    max_a (A x_2)(a) - min_b (x_1^T A)(b), where A = U_1. Zero exactly at a
    Nash equilibrium.
    """
    a = utilities[0]
    return jnp.max(a @ x[1]) - jnp.min(x[0] @ a)
