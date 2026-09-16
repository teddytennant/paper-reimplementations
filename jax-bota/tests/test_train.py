"""Learning-method smoke: L_BFM decreases on a 2D Gaussian-mixture pair."""

import jax.numpy as jnp
from jax import random

from jax_bota.train import sample_gmm_pair, train_bfm, train_bota


def test_gmm_pair_shapes():
    x0, x1 = sample_gmm_pair(random.PRNGKey(0), n=10, dim=2)
    assert x0.shape == (10, 2)
    assert x1.shape == (10, 2)


def test_bfm_loss_decreases():
    key = random.PRNGKey(2)
    x0, x1 = sample_gmm_pair(key, n=12, dim=2)
    result = train_bota(
        x0,
        x1,
        alpha=0.5,
        n_time=4,
        n_discrete_opt=8,
        n_bfm_steps=80,
        hidden=24,
        lr_bfm=8e-3,
        seed=2,
    )
    losses = result.losses
    assert losses.shape[0] == 80
    start = float(jnp.mean(losses[:8]))
    end = float(jnp.mean(losses[-8:]))
    assert end < start


def test_train_bfm_on_straight_paths_decreases():
    from jax_bota.particles import euler_trajectories, straight_velocities

    x0 = jnp.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, -0.7], [1.0, -0.7]])
    x1 = x0 + jnp.array([0.0, 2.0])
    v = straight_velocities(x0, x1, 4)
    x = euler_trajectories(x0, v, 0.25)
    _, losses = train_bfm(x, v, n_steps=60, hidden=16, lr=1e-2, batch=32, seed=3)
    assert float(losses[-1]) < float(losses[0])
