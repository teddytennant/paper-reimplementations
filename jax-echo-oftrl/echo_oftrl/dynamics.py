"""ECHO-OFTRL self-play, Algorithm 1 of arXiv:2608.31166.

Each round, player i forms the predictor ghat_i from its own cascade state
(3.4), plays Q_i(eta (S_i + ghat_i)) (3.5), observes u_i, forms
g_i = u_i - <u_i, x_i> 1, updates the cascade by (3.6), and adds g_i to S_i.
Nothing in that list touches another player's state, which is what "uncoupled"
means in the paper and what `replay_from_gradients` below is here to exhibit.

External regret comes out of S_i for free. Since
g_i^(t)(a) = u_i^(t)(a) - <u_i^(t), x_i^(t)>,

    Reg_i(T) = max_x sum_t <u_i^(t), x - x_i^(t)> = max_a S_i^(T)(a),

so the regret trace is a running max over actions of the accumulated shifted
payoffs, and every round contributes an O(1) quantity instead of a difference
of two O(T) sums.

Setting depth=1 and rho=0.0 collapses the cascade to ghat^(t) = g^(t-1), which
is ordinary one-step-recency optimism on the same lifted geometry. That is the
comparison baseline.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cascade import ema_cascade_predict, ema_cascade_step
from .games import payoff_vectors, shifted_gradients
from .lifted import gamma_const, lifted_argmax

# Theorem 3.3 sets eta = 2^-94 / (N^20 Xi_max), which is around 1e-40 and does
# nothing in finite time. This is the largest round number that stayed flat on
# every game in tests/test_dynamics.py; at eta = 2 the regret on biased
# matching pennies starts growing again.
DEFAULT_ETA = 1.0


class Trajectory(NamedTuple):
    """Per-round record of a run. T is the horizon, N players, m actions."""

    regret: jnp.ndarray  # (T, N), regret[t-1, i] = Reg_i(t)
    strategies: jnp.ndarray  # (T, N, m), the played x_i^(t)
    lambdas: jnp.ndarray  # (T, N), the lifted scale lam_i^(t)
    gradients: jnp.ndarray  # (T, N, m), the shifted g_i^(t)

    @property
    def final_regret(self) -> jnp.ndarray:
        """Reg_i(T) for every i."""
        return self.regret[-1]

    @property
    def mean_strategy(self) -> jnp.ndarray:
        """The average iterate, (N, m)."""
        return jnp.mean(self.strategies, axis=0)


def cascade_params(num_players: int) -> tuple[float, int]:
    """rho = 1 - 1/N and depth N, the setting of equation (3.2)."""
    return 1.0 - 1.0 / num_players, num_players


def run_game(
    utilities: jnp.ndarray,
    horizon: int,
    eta: float = DEFAULT_ETA,
    rho: float | None = None,
    depth: int | None = None,
) -> Trajectory:
    """Run all N players under Algorithm 1 for `horizon` rounds.

    `utilities` has shape (N,) + (m,)*N with entries in [0,1]. `rho` and
    `depth` default to the paper's 1 - 1/N and N.
    """
    num_players = utilities.shape[0]
    num_actions = utilities.shape[1]
    if rho is None or depth is None:
        default_rho, default_depth = cascade_params(num_players)
        rho = default_rho if rho is None else rho
        depth = default_depth if depth is None else depth
    gamma = gamma_const(num_actions)
    dtype = utilities.dtype

    init = (
        jnp.zeros((num_players, num_actions), dtype),
        jnp.zeros((num_players, depth, num_actions), dtype),
    )

    def step(carry, _):
        acc, ema = carry
        ghat = ema_cascade_predict(ema)
        lam, x = lifted_argmax(eta * (acc + ghat), gamma)
        u = payoff_vectors(utilities, x)
        g = shifted_gradients(u, x)
        acc = acc + g
        ema, _ = ema_cascade_step(ema, g, rho)
        return (acc, ema), (jnp.max(acc, axis=-1), x, lam, g)

    _, out = jax.lax.scan(step, init, None, length=horizon)
    return Trajectory(*out)


def replay_from_gradients(
    gradients: jnp.ndarray,
    num_actions_gamma: float,
    eta: float = DEFAULT_ETA,
    rho: float = 0.5,
    depth: int = 2,
) -> jnp.ndarray:
    """Rerun updates against a fixed gradient sequence, with the game gone.

    `gradients` is (T, m) for a single player, or (T, N, m) to drive N players
    side by side in the same shapes `run_game` carries. The returned strategies
    have the matching shape. Feeding a player's own g_i sequence from `run_game`
    back through here reproduces its iterates, since the only inputs to lines
    3-5 are S_i and the cascade state.

    Driving N players at once is not the same as running the game: the players
    never see each other here, each row is its own recursion over its own
    column of `gradients`. That is what makes the (T, N, m) form the useful
    uncoupledness probe. Change one row's gradients and no other row may move.
    """
    batch = gradients.shape[1:-1]
    m = gradients.shape[-1]
    init = (
        jnp.zeros(batch + (m,), gradients.dtype),
        jnp.zeros(batch + (depth, m), gradients.dtype),
    )

    def step(carry, g):
        acc, ema = carry
        _, x = lifted_argmax(eta * (acc + ema_cascade_predict(ema)), num_actions_gamma)
        ema, _ = ema_cascade_step(ema, g, rho)
        return (acc + g, ema), x

    _, xs = jax.lax.scan(step, init, gradients)
    return xs
