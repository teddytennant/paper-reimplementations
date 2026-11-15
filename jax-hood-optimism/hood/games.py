"""N-player normal form games with K actions per player, Section 2.

A game is stored as one payoff tensor per player, stacked into an array of
shape (N,) + (K,) * N, so ``payoffs[i][a_0, ..., a_{N-1}]`` is u_i(a) in the
notation of Section 2.1. Payoffs live in [-1, 1], which is what the centering
bound (13) assumes.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class NormalFormGame(NamedTuple):
    """Finite N-player game in normal form, Section 2.1.

    Attributes:
        payoffs: array of shape (N,) + (K,) * N with entries in [-1, 1].
    """

    payoffs: jnp.ndarray

    @property
    def num_players(self) -> int:
        return self.payoffs.shape[0]

    @property
    def num_actions(self) -> int:
        return self.payoffs.shape[1]


def payoff_vectors(game: NormalFormGame, x: jnp.ndarray) -> jnp.ndarray:
    """Full-information feedback of eq. (9): v_{i,alpha} = u_i(alpha, x_{-i}).

    Args:
        game: the game.
        x: mixed profile of shape (N, K), each row on the simplex.

    Returns:
        Array of shape (N, K). Row i is player i's payoff vector v_i.
    """
    n = game.num_players
    rows = []
    for i in range(n):
        # Put player i's own axis first, then contract every other axis
        # against that player's mixed strategy.
        tensor = jnp.moveaxis(game.payoffs[i], i, 0)
        for j in reversed([j for j in range(n) if j != i]):
            tensor = jnp.tensordot(tensor, x[j], axes=([tensor.ndim - 1], [0]))
        rows.append(tensor)
    return jnp.stack(rows)


def utilities(game: NormalFormGame, x: jnp.ndarray) -> jnp.ndarray:
    """u_i(x) for every player, eq. (10)."""
    return jnp.sum(x * payoff_vectors(game, x), axis=-1)


def centered_payoffs(game: NormalFormGame, x: jnp.ndarray) -> jnp.ndarray:
    """The centered payoff vectors g_i = v_i - u_i(x) 1 of eq. (11).

    Satisfies <x_i, g_i> = 0 and ||g_i||_inf <= 2, eq. (13).
    """
    v = payoff_vectors(game, x)
    u = jnp.sum(x * v, axis=-1)
    return v - u[:, None]


def random_game(key: jax.Array, num_players: int, num_actions: int) -> NormalFormGame:
    """General-sum game with payoffs drawn uniformly from [-1, 1]."""
    shape = (num_players,) + (num_actions,) * num_players
    return NormalFormGame(jax.random.uniform(key, shape, minval=-1.0, maxval=1.0))


def matching_pennies() -> NormalFormGame:
    """Two-player zero-sum matching pennies."""
    row = jnp.array([[1.0, -1.0], [-1.0, 1.0]])
    return NormalFormGame(jnp.stack([row, -row]))


def coordination_game(num_players: int = 2, num_actions: int = 3) -> NormalFormGame:
    """Everyone gets 1 when all players pick the same action, -1 otherwise."""
    shape = (num_actions,) * num_players
    idx = jnp.indices(shape)
    agree = jnp.all(idx == idx[0], axis=0)
    common = jnp.where(agree, 1.0, -1.0)
    return NormalFormGame(jnp.stack([common] * num_players))


def rock_paper_scissors() -> NormalFormGame:
    """Two-player zero-sum RPS, a convenient cycling instance."""
    row = jnp.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    return NormalFormGame(jnp.stack([row, -row]))
