"""Mass conservation, Euler step, and discrete-time solver."""

import jax.numpy as jnp
import numpy as np
from jax import random

from jax_bota.particles import (
    discrete_loss,
    euler_trajectories,
    normalize_masses,
    solve_discrete,
    straight_velocities,
    terminal_mse,
)


def test_masses_nonnegative_and_conserved():
    raw = jnp.array([0.2, -0.1, 0.5, 0.4])
    masses = normalize_masses(raw, total=1.0)
    assert float(jnp.min(masses)) >= 0.0
    np.testing.assert_allclose(float(jnp.sum(masses)), 1.0, rtol=1e-6)


def test_euler_reaches_target_on_straight_path():
    x0 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    x1 = jnp.array([[0.0, 2.0], [1.0, 2.0]])
    t_steps = 5
    dt = 1.0 / t_steps
    v = straight_velocities(x0, x1, t_steps)
    xs = euler_trajectories(x0, v, dt)
    assert xs.shape == (t_steps + 1, 2, 2)
    np.testing.assert_allclose(np.array(xs[0]), np.array(x0), atol=1e-6)
    np.testing.assert_allclose(np.array(xs[-1]), np.array(x1), atol=1e-5)


def test_solve_discrete_preserves_mass():
    key = random.PRNGKey(0)
    x0 = random.normal(key, (6, 2)) * 0.2 + jnp.array([-0.5, 0.0])
    x1 = x0 + jnp.array([0.0, 1.0])
    masses = jnp.full((6,), 1.0 / 6.0)
    result = solve_discrete(
        x0, x1, alpha=0.5, n_steps=4, n_opt=5, lr=0.05, masses=masses
    )
    assert float(jnp.min(result.masses)) >= 0.0
    np.testing.assert_allclose(float(jnp.sum(result.masses)), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.array(result.masses), np.array(masses), atol=1e-6)


def test_solve_discrete_reduces_terminal_error():
    x0 = jnp.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, -0.8], [1.0, -0.8]])
    x1 = jnp.array([[-1.0, 1.0], [1.0, 1.0], [-1.0, 0.8], [1.0, 0.8]])
    v0 = jnp.zeros((4, 4, 2))
    dt = 0.25
    from jax_bota.cost import bounding_grid_centers

    centers = bounding_grid_centers(jnp.concatenate([x0, x1], axis=0), n_per_axis=3)
    loss0 = discrete_loss(v0, x0, x1, centers, alpha=0.5, dt=dt, lam=5.0)
    result = solve_discrete(x0, x1, alpha=0.5, n_steps=4, n_opt=30, lr=0.1, lam=5.0)
    term0 = terminal_mse(x0, x1)
    term1 = terminal_mse(result.positions[-1], x1)
    assert float(term1) < float(term0)
    assert float(result.loss) <= float(loss0) + 1e-4
