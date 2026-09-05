"""Algorithm 1 and Algorithm 2 on games small enough to check by hand."""

import jax
import jax.numpy as jnp
import numpy as np

from layered_omd import algorithm, games, gap, values
from layered_omd.mirror import prox_step, step_sizes


def _biased_pennies_init():
    return (
        jnp.array([[[0.9, 0.1]]]),
        jnp.array([[[0.2, 0.8]]]),
    )


def test_uniform_is_a_fixed_point_of_matching_pennies():
    """The unique NE is uniform, and Algorithm 1 starts there (line 3)."""
    game = games.matching_pennies()
    traj = algorithm.run_full_feedback(game, 25, eta0=0.5)
    for i in range(2):
        assert np.max(np.abs(np.asarray(traj[i]) - 0.5)) < 1e-12


def test_full_feedback_cce_gap_shrinks_on_matching_pennies():
    """Definition 4: the empirical CCE gap of the whole trajectory is below the prefix.

    Matching pennies has a unique CCE at uniform. Starting away from it, the
    average of T=60 played profiles has less unilateral deviation value than
    the first eight iterates on their own.
    """
    game = games.matching_pennies()
    traj = algorithm.run_full_feedback(game, 60, eta0=0.5, init=_biased_pennies_init())
    early = gap.cce_gap(game, tuple(a[:8] for a in traj))
    late = gap.cce_gap(game, traj)
    assert early > 0.0
    assert late < early
    assert late < 0.12


def test_one_layer_played_iterate_is_the_optimistic_prox():
    """H = 1: x_t = P_{x_tilde_{t-1}}(Q_{t-1}) and x_tilde_t = P_{x_tilde_{t-1}}(Q_t)."""
    game = games.matching_pennies()
    init = _biased_pennies_init()
    n_episodes = 6
    traj = algorithm.run_full_feedback(game, n_episodes, eta0=0.5, init=init)
    eta = step_sizes(game.horizon, n_episodes, 0.5)
    lam = algorithm.smoothing(game)
    xt = init
    q_prev = tuple(jnp.zeros((1, 1, n)) for n in game.n_actions)
    for t in range(n_episodes):
        played = tuple(
            prox_step(q_prev[i], xt[i], eta[:, None], lam[i], 0.0) for i in range(2)
        )
        for i in range(2):
            assert np.max(np.abs(np.asarray(traj[i][t] - played[i]))) < 1e-12
        q_now = tuple(values.player_values(game, played, i)[0] for i in range(2))
        xt = tuple(
            prox_step(q_now[i], xt[i], eta[:, None], lam[i], 0.0) for i in range(2)
        )
        q_prev = q_now


def test_partial_feedback_estimate_matches_exact_q_on_a_matrix_game():
    """H = 1, every (s, a_i) is visited, so Qhat is the sample mean of the cost."""
    game = games.matching_pennies()
    policies = (
        jnp.array([[[0.35, 0.65]]]),
        jnp.array([[[0.55, 0.45]]]),
    )
    exact = [np.asarray(values.player_values(game, policies, i)[0]) for i in range(2)]
    rollout = algorithm.sample_block(game, policies, 20000, jax.random.key(4))
    est = [np.asarray(q) for q in algorithm.estimate_q(game, rollout)]
    for i in range(2):
        assert np.max(np.abs(est[i] - exact[i])) < 0.03
