"""ECHO-OFTRL self-play: the claims of Theorem 3.3 that a small run can check."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from echo_oftrl import (
    cascade_params,
    duality_gap,
    gamma_const,
    lifted_argmax,
    matching_pennies,
    payoff_vectors,
    random_game,
    replay_from_gradients,
    rock_paper_scissors,
    run_game,
)
from echo_oftrl.games import biased_matching_pennies

ETA = 1.0
CHECKPOINTS = (200, 400, 800, 1600)


def _suite():
    """Eight small games: one 2x2 constant sum, two 4x4 constant sum, five general sum."""
    rng = np.random.default_rng(7)
    games = {"biased_matching_pennies": biased_matching_pennies()}
    for s in range(2):
        a = rng.random((4, 4))
        games[f"zero_sum_4x4_seed{s}"] = jnp.asarray(np.stack([a, 1.0 - a]))
    for s in (0, 1, 2):
        games[f"general_3p_seed{s}"] = random_game(3, 3, s)
    for s in (10, 11):
        games[f"general_4p_seed{s}"] = random_game(4, 2, s)
    return games


SUITE = _suite()
_RUNS = {}


def _run(name, horizon, **kwargs):
    """run_game with a cache, since the same trajectory answers several tests."""
    key = (name, horizon, tuple(sorted(kwargs.items())))
    if key not in _RUNS:
        _RUNS[key] = run_game(SUITE[name], horizon, eta=ETA, **kwargs)
    return _RUNS[key]


def _max_regret_trace(traj):
    return np.asarray(traj.regret).max(axis=1)


@pytest.mark.parametrize("name", sorted(SUITE))
def test_iterates_stay_in_the_simplex(name):
    traj = _run(name, 400)
    x = np.asarray(traj.strategies)
    assert x.min() > 0.0
    assert np.abs(x.sum(axis=-1) - 1.0).max() < 1e-14
    lam = np.asarray(traj.lambdas)
    assert lam.min() > 0.0 and lam.max() < 1.0


@pytest.mark.parametrize("name", sorted(SUITE))
def test_regret_trace_matches_the_definition(name):
    """Reg_i(T) = max_x sum_t <u_i^(t), x - x_i^(t)>, Section 2.1.

    run_game reads this off S_i, which accumulates g_i = u_i - <u_i, x_i> 1 one
    round at a time. Here it is recomputed the literal way, as a difference of
    two sums of size T, so the two disagree only by float64 rounding over 400
    rounds.
    """
    utilities = SUITE[name]
    traj = _run(name, 400)
    u = jax.vmap(payoff_vectors, in_axes=(None, 0))(utilities, traj.strategies)
    played = jnp.sum(u * traj.strategies, axis=-1).sum(axis=0)
    direct = jnp.max(u.sum(axis=0), axis=-1) - played
    assert np.allclose(direct, traj.final_regret, rtol=0, atol=1e-11)


@pytest.mark.parametrize("name", sorted(SUITE))
def test_individual_regret_is_horizon_independent(name):
    """Theorem 3.3's claim: a bound on max_i Reg_i(T) holding for every T at once.

    One run to T = 1600 gives Reg_i(t) for every t, since the prefix of a
    deterministic run is the run. The check is that nothing after t = 200 rises
    more than 0.2 above the value at t = 200. Growth of order sqrt(T) would need
    a factor 2.83 over that range and log T a factor 1.41, both far outside the
    slack for every game here, where Reg_i(200) is between 0.24 and 1.29.
    """
    trace = _max_regret_trace(_run(name, max(CHECKPOINTS)))
    base = trace[CHECKPOINTS[0] - 1]
    assert trace[CHECKPOINTS[0] - 1 :].max() <= base + 0.2
    assert trace[-1] <= base + 0.2


@pytest.mark.parametrize("name", sorted(SUITE))
def test_sum_of_positive_regrets_is_horizon_independent(name):
    """The left-hand side of Theorem 3.3 is sum_i [Reg_i(T)]_+, not max_i Reg_i(T)."""
    traj = _run(name, max(CHECKPOINTS))
    positive = np.clip(np.asarray(traj.regret), 0.0, None).sum(axis=1)
    assert positive[CHECKPOINTS[0] - 1 :].max() <= positive[CHECKPOINTS[0] - 1] + 0.5


@pytest.mark.parametrize("name", ["biased_matching_pennies", "general_3p_seed1"])
def test_optimism_is_what_removes_the_growth(name):
    """A control that separates: the same code with the predictor switched off.

    depth 1 and rho 1 pins the cascade state at zero, so ghat = 0 and the update
    is lifted FTRL with no optimism at all. On these two games its regret at
    t = 1600 is at least 2x its regret at t = 200 while ECHO's is flat, which is
    what says the flatness above comes from the predictor and not from the game
    being easy.
    """
    echo = _max_regret_trace(_run(name, 1600))
    plain = _max_regret_trace(_run(name, 1600, depth=1, rho=1.0))
    assert plain[1599] >= 2.0 * plain[199]
    assert echo[1599] <= echo[199] + 0.2
    assert echo[1599] < plain[1599]


def test_average_iterate_converges_to_the_known_nash():
    """Biased matching pennies, payoffs [[1, 0], [0, 1/2]], equilibrium (1/3, 2/3)."""
    utilities = biased_matching_pennies()
    traj = run_game(utilities, 2000, eta=ETA)
    mean = np.asarray(traj.mean_strategy)
    assert np.allclose(mean, np.array([[1 / 3, 2 / 3], [1 / 3, 2 / 3]]), atol=1e-3)
    assert float(duality_gap(utilities, traj.mean_strategy)) < 5e-4


@pytest.mark.parametrize("horizon", [200, 800, 2000])
def test_duality_gap_is_under_the_regret_sum_over_horizon(horizon):
    """gap(average iterate) <= sum_i Reg_i(T) / T in a two-player constant-sum game.

    The bound comes out tight here, within 1e-16 at T = 2000, so the slack is
    1e-12 and not a percentage.
    """
    utilities = biased_matching_pennies()
    traj = run_game(utilities, horizon, eta=ETA)
    gap = float(duality_gap(utilities, traj.mean_strategy))
    assert gap <= float(traj.final_regret.sum()) / horizon + 1e-12
    assert gap < 5.0 / horizon


@pytest.mark.parametrize("player", [0, 1, 2])
def test_updates_use_only_the_players_own_gradients(player):
    """Uncoupledness as a property of the code, not a comment.

    Lines 3 to 5 of Algorithm 1 read only S_i and ema_i, both of which are
    functions of g_i alone. Drive all three players through the replay driver
    at once, then again with the other two players' gradient streams reversed
    and negated. This player's iterates have to come back bit for bit, and the
    other two have to move.

    Perturbing an input row rather than comparing against a second program is
    what makes bit equality the right assertion here. It is one compiled
    function, called twice on the same shapes, so anything that survives is a
    statement about the dataflow: a sum over the player axis instead of the
    cascade axis, a shared buffer, a transposed vmap, all of them move this
    row. The last line is there so the probe cannot pass by doing nothing.
    """
    utilities = random_game(3, 3, 1)
    traj = run_game(utilities, 300, eta=ETA)
    rho, depth = cascade_params(3)
    driver = dict(eta=ETA, rho=rho, depth=depth)
    base = np.asarray(replay_from_gradients(traj.gradients, gamma_const(3), **driver))

    others = [j for j in range(3) if j != player]
    perturbed = np.asarray(traj.gradients).copy()
    perturbed[:, others] = -perturbed[::-1][:, others]
    moved = np.asarray(replay_from_gradients(jnp.asarray(perturbed), gamma_const(3), **driver))

    assert np.array_equal(moved[:, player], base[:, player])
    assert np.abs(moved[:, others] - base[:, others]).max() > 0.1


@pytest.mark.parametrize("driven", ["one_player_at_a_time", "all_players_at_once"])
def test_replay_of_the_gradients_reproduces_the_run(driven):
    """The whole run rebuilt from its own g, with the game deleted.

    Same recursion as `run_game`, but `run_game` fuses it with the payoff
    tensordots and the replay driver has nothing to fuse with, so XLA is free
    to order the length-m reductions in Q_m differently between the two. On CPU
    it does not and the two agree exactly; on an H200 the softmax, logsumexp
    and <theta, x> inside the bisection come out up to one ulp apart, which
    moved 8 of one player's 300 rounds by 2.2e-16. atol 1e-14 is 45 ulp at
    this scale, far under anything a real coupling bug would produce, since
    that would put one player's whole gradient into another player's theta.
    """
    utilities = random_game(3, 3, 1)
    traj = run_game(utilities, 300, eta=ETA)
    rho, depth = cascade_params(3)
    driver = dict(eta=ETA, rho=rho, depth=depth)
    if driven == "all_players_at_once":
        replayed = replay_from_gradients(traj.gradients, gamma_const(3), **driver)
    else:
        replayed = jnp.stack(
            [replay_from_gradients(traj.gradients[:, i], gamma_const(3), **driver) for i in range(3)],
            axis=1,
        )
    assert np.allclose(np.asarray(replayed), np.asarray(traj.strategies), rtol=0, atol=1e-14)


@pytest.mark.parametrize("utilities", [matching_pennies(), rock_paper_scissors()])
def test_uniform_equilibrium_is_a_fixed_point(utilities):
    """Both games have the uniform profile as their unique equilibrium.

    Q_i(0) is uniform, so g_i is exactly zero at round 1 and the dynamics never
    leave. Regret is exactly zero, not approximately.
    """
    traj = run_game(utilities, 300, eta=ETA)
    assert np.abs(np.asarray(traj.gradients)).max() == 0.0
    assert np.abs(np.asarray(traj.final_regret)).max() == 0.0
    m = utilities.shape[1]
    assert np.allclose(np.asarray(traj.strategies), 1.0 / m, rtol=0, atol=1e-15)


def _algorithm_1_by_hand(utilities, horizon, eta, rho, depth):
    """Algorithm 1 transcribed line by line with a Python loop, for comparison."""
    n, m = utilities.shape[0], utilities.shape[1]
    gamma = gamma_const(m)
    acc = np.zeros((n, m))
    ema = np.zeros((n, depth, m))
    strategies = []
    for _ in range(horizon):
        ghat = ema.sum(axis=1)  # line 3
        x = np.stack(
            [np.asarray(lifted_argmax(jnp.asarray(eta * (acc[i] + ghat[i])), gamma)[1]) for i in range(n)]
        )  # lines 4-5
        u = np.asarray(payoff_vectors(utilities, jnp.asarray(x)))
        g = u - (u * x).sum(axis=1, keepdims=True)  # line 5
        e = g.copy()  # e_{i,0} = g_i
        new_ema = np.zeros_like(ema)
        for h in range(depth):  # lines 6-8
            new_ema[:, h] = rho * ema[:, h] + (1.0 - rho) * e
            e = e - ema[:, h]
        ema = new_ema
        acc = acc + g  # line 9
        strategies.append(x)
    return np.asarray(strategies)


@pytest.mark.parametrize("num_players", [2, 3])
def test_scan_matches_algorithm_1_written_as_a_loop(num_players):
    """The lax.scan and prefix-sum version against a literal transcription."""
    utilities = random_game(num_players, 3, seed=21)
    rho, depth = cascade_params(num_players)
    reference = _algorithm_1_by_hand(utilities, 60, ETA, rho, depth)
    traj = run_game(utilities, 60, eta=ETA)
    assert np.allclose(reference, np.asarray(traj.strategies), rtol=0, atol=1e-13)
