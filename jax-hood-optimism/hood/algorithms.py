"""Algorithm 1 (HOOD) and the OptFTRL baseline it is compared against.

Algorithm 1, verbatim:

    1: y_{i,0} <- 0, g_{i,k}, m_{i,k} <- 0 for -N <= k <= 0
    2: for t = 1, 2, ... do
    3:   compute m_{i,t} from (22), set y_{i,t-1/2} <- y_{i,t-1} + m_{i,t}
    4:   update x_{i,t} <- Q(eta y_{i,t-1/2}), play x_{i,t}
    5:   observe v_{i,t}, set g_{i,t} <- v_{i,t} - u_i(x_t) 1 and y_{i,t} <- y_{i,t-1} + g_{i,t}
    6: end for

All players run it in self-play, which is the setting of Theorem 1. The loop is
a single ``lax.scan`` over rounds with the lag buffers in the carry.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from hood.choice import DEFAULT_ITERS, entropic_choice, lifted_choice, regularizer_constant
from hood.games import NormalFormGame, centered_payoffs
from hood.predictors import first_order_predictor, hood_predictor, push

Predictor = Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray]
ChoiceMap = Callable[[jnp.ndarray, float], jnp.ndarray]


class Trajectory(NamedTuple):
    """Per-round record of a run.

    Attributes:
        x: mixed profiles x_t, shape (T, N, K).
        g: centered payoff vectors g_t of eq. (11), shape (T, N, K).
        m: predictions m_t, shape (T, N, K).
    """

    x: jnp.ndarray
    g: jnp.ndarray
    m: jnp.ndarray

    @property
    def prediction_error(self) -> jnp.ndarray:
        """e_t = g_t - m_t, eq. (24)."""
        return self.g - self.m


def _run(
    game: NormalFormGame,
    eta: float,
    rounds: int,
    predictor: Predictor,
    choice_map: ChoiceMap,
) -> Trajectory:
    """Shared driver for Algorithm 1 and its first-order ablation."""
    n, k = game.num_players, game.num_actions
    L = regularizer_constant(k)
    order = n + 1

    def play(y_half: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda row: choice_map(row, L))(eta * y_half)

    def step(carry, _):
        y, g_buf, m_buf = carry
        # Algorithm 1, line 3.
        m_t = predictor(g_buf, m_buf, n)
        y_half = y + m_t
        # Algorithm 1, line 4.
        x_t = play(y_half)
        # Algorithm 1, line 5. centered_payoffs reads v_{i,t} off the game and
        # subtracts the realized utility, which is eq. (11).
        g_t = centered_payoffs(game, x_t)
        y = y + g_t
        return (y, push(g_buf, g_t), push(m_buf, m_t)), (x_t, g_t, m_t)

    # Algorithm 1, line 1.
    init = (
        jnp.zeros((n, k)),
        jnp.zeros((order, n, k)),
        jnp.zeros((order, n, k)),
    )
    _, (x, g, m) = jax.lax.scan(step, init, None, length=rounds)
    return Trajectory(x, g, m)


def run_hood(
    game: NormalFormGame,
    eta: float,
    rounds: int,
    lam_iters: int = DEFAULT_ITERS,
) -> Trajectory:
    """Run Algorithm 1 (HOOD) in self-play.

    Args:
        game: the game every player learns in.
        eta: step size. Theorem 1 needs 0 < eta <= 1 / (60 (N+1)^3), eq. (26).
        rounds: horizon T. The algorithm itself is horizon-free.
        lam_iters: golden-section iterations inside the choice map of eq. (21).

    Returns:
        The trajectory of play.
    """
    choice = lambda y, L: lifted_choice(y, L, lam_iters)  # noqa: E731
    return _run(game, eta, rounds, hood_predictor, choice)


def run_optftrl(
    game: NormalFormGame,
    eta: float,
    rounds: int,
    lifted: bool = False,
    lam_iters: int = DEFAULT_ITERS,
) -> Trajectory:
    """Baseline: OptFTRL with the one-step predictor m_t = g_{t-1}.

    With ``lifted=False`` this is optimistic multiplicative weights (OMWU),
    the standard entropic choice map softmax(eta y). With ``lifted=True`` it
    keeps the lifted map of eq. (20) and changes only the predictor, which
    isolates the contribution of the higher-order prediction of eq. (22).
    """
    if lifted:
        choice = lambda y, L: lifted_choice(y, L, lam_iters)  # noqa: E731
    else:
        choice = entropic_choice
    return _run(game, eta, rounds, first_order_predictor, choice)


run_hood_jit = jax.jit(run_hood, static_argnames=("rounds", "lam_iters"))
run_optftrl_jit = jax.jit(run_optftrl, static_argnames=("rounds", "lifted", "lam_iters"))
