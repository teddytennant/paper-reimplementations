"""Theorem 1 sanity checks on tiny self-play instances."""

import jax
import jax.numpy as jnp
import numpy as np

from hood.algorithms import run_hood
from hood.games import matching_pennies, random_game
from hood.regret import (
    cce_gap,
    empirical_distribution,
    individual_regret,
    max_step_size,
    regret_bound,
)


def test_theorem1_matching_pennies():
    game = matching_pennies()
    eta = max_step_size(2)
    traj = run_hood(game, eta, 80)
    regret = individual_regret(traj.g)
    bound = regret_bound(2, eta)
    assert float(jnp.max(regret)) <= bound + 1e-8

    assert traj.x.shape == (80, 2, 2)
    assert traj.g.shape == (80, 2, 2)
    assert traj.m.shape == (80, 2, 2)

    np.testing.assert_allclose(np.asarray(traj.x.sum(axis=-1)), 1.0, atol=1e-8)
    assert np.all(np.asarray(traj.x) >= -1e-12)

    mix = np.asarray(jnp.mean(traj.x, axis=0))
    np.testing.assert_allclose(mix, 0.5, atol=0.25)

    gap = cce_gap(game, empirical_distribution(traj.x))
    assert np.all(np.isfinite(np.asarray(gap)))


def test_theorem1_random_game():
    game = random_game(jax.random.key(0), 2, 3)
    eta = max_step_size(2)
    traj = run_hood(game, eta, 40)
    regret = individual_regret(traj.g)
    bound = regret_bound(3, eta)
    assert float(jnp.max(regret)) <= bound + 1e-8
    assert traj.x.shape == (40, 2, 3)
    np.testing.assert_allclose(np.asarray(traj.x.sum(axis=-1)), 1.0, atol=1e-8)
