"""Individual regret, the Theorem 1 bound, and the CCE gap of Corollary 1."""

import jax.numpy as jnp

from hood.choice import regularizer_constant
from hood.games import NormalFormGame


def individual_regret(g: jnp.ndarray) -> jnp.ndarray:
    """Reg_i(t) for every prefix t, eq. (14).

    Reg_i(T) = max_alpha sum_{t<=T} g_{i,alpha,t}, and the centering of
    eq. (11) is what makes this equal to the usual external regret.

    Args:
        g: centered payoff vectors, shape (T, N, K).

    Returns:
        Array of shape (T, N) with the regret at every prefix, so that the
        supremum over T of Theorem 1 can be checked and not just its last term.
    """
    return jnp.max(jnp.cumsum(g, axis=0), axis=-1)


def regret_bound(num_actions: int, eta: float) -> float:
    """The Theorem 1 bound 3L / eta, eq. (27)."""
    return 3.0 * regularizer_constant(num_actions) / eta


def max_step_size(num_players: int) -> float:
    """The largest step size Theorem 1 admits, 1 / (60 (N+1)^3), eq. (26)."""
    return 1.0 / (60.0 * (num_players + 1.0) ** 3)


def empirical_distribution(x: jnp.ndarray) -> jnp.ndarray:
    """mu_T = T^{-1} sum_t prod_i x_{i,t}, the measure in Corollary 1.

    Args:
        x: mixed profiles, shape (T, N, K).

    Returns:
        Joint distribution over action profiles, shape (K,) * N.
    """
    rounds, n, k = x.shape
    joint = x[:, 0]
    for i in range(1, n):
        joint = joint[..., None] * x[:, i].reshape((rounds,) + (1,) * i + (k,))
    return jnp.mean(joint, axis=0)


def cce_gap(game: NormalFormGame, mu: jnp.ndarray) -> jnp.ndarray:
    """Coarse correlated equilibrium gap of a joint distribution.

    max over deviations alpha of E_{a~mu}[u_i(alpha, a_{-i})] - E_{a~mu}[u_i(a)],
    per player. mu is an eps-CCE when this is at most eps for every i, which is
    the sense in which Corollary 1 calls mu_T a (3L / (eta T))-approximate CCE.

    Args:
        game: the game.
        mu: joint distribution, shape (K,) * N.

    Returns:
        Array of shape (N,) with each player's gap.
    """
    n = game.num_players
    gaps = []
    for i in range(n):
        payoff = game.payoffs[i]
        on_path = jnp.sum(mu * payoff)
        # Marginal of mu over the opponents, then the deviation payoffs.
        others = jnp.sum(mu, axis=i)
        tensor = jnp.moveaxis(payoff, i, 0)
        deviate = jnp.tensordot(tensor, others, axes=(list(range(1, n)), list(range(n - 1))))
        gaps.append(jnp.max(deviate) - on_path)
    return jnp.stack(gaps)
