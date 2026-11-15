"""Sec. II-D: mean first, sparse neighbour last, as sigma comes down."""

import jax
import jax.numpy as jnp
import pytest

from diffusion_is_attention import (
    attention_weights,
    figure2_setup,
    reverse_attention_trajectory,
    sigma_schedule,
    weight_entropy,
)


def test_sigma_schedule_matches_figure_2():
    sigmas = sigma_schedule()
    assert float(sigmas[0]) == pytest.approx(1.0)
    assert float(sigmas[-1]) == pytest.approx(0.37, abs=0.01)
    ratios = sigmas[1:] / sigmas[:-1]
    assert jnp.allclose(ratios, 0.96)


def test_figure2_trajectory_visits_the_mean_then_a_signal():
    data, y, mean = figure2_setup()
    traj = reverse_attention_trajectory(y, data, sigma_schedule())
    to_mean = jnp.linalg.norm(traj.points - mean, axis=-1)

    # It passes near M early, well before it settles.
    early = jnp.argmin(to_mean)
    assert int(early) < 6
    assert float(to_mean[early]) < 0.1 * float(to_mean[0])

    # It ends on one of the three signals, not between them.
    final = traj.points[-1]
    gaps = jnp.linalg.norm(data - final, axis=-1)
    assert float(jnp.min(gaps)) < 0.02
    assert float(jnp.min(gaps)) < 0.05 * float(jnp.median(gaps))

    # And it leaves the mean behind.
    assert float(to_mean[-1]) > 0.5


def test_figure2_weights_sparsify():
    data, y, _ = figure2_setup()
    traj = reverse_attention_trajectory(y, data, sigma_schedule())
    h = weight_entropy(traj.weights)
    uniform = jnp.log(data.shape[0])
    assert float(h[0]) > 0.9 * float(uniform)  # large sigma, nearly uniform
    assert float(h[-1]) < 0.05 * float(uniform)  # small sigma, nearly one-hot
    assert float(h[-1]) < float(h[0])
    # No excursion back to a flatter weighting once it starts concentrating.
    assert jnp.all(jnp.diff(h[2:]) < 1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_entropy_is_monotone_in_sigma_at_a_fixed_query(seed):
    """At fixed y the weights are a Gibbs distribution at temperature sigma^2."""
    k1, k2 = jax.random.split(jax.random.key(seed))
    data = jax.random.normal(k1, (24, 5))
    y = jax.random.normal(k2, (5,))
    sigmas = jnp.geomspace(0.05, 20.0, 60)
    h = jnp.array([weight_entropy(attention_weights(y, data, s)) for s in sigmas])
    assert jnp.all(jnp.diff(h) > -1e-12)
    assert float(h[0]) < 1e-6
    assert float(h[-1]) == pytest.approx(float(jnp.log(24)), abs=1e-3)


def test_trajectory_shapes():
    data, y, _ = figure2_setup()
    sigmas = sigma_schedule(num_steps=7)
    traj = reverse_attention_trajectory(y, data, sigmas)
    assert traj.points.shape == (8, 2)
    assert traj.weights.shape == (7, 3)
    assert jnp.allclose(traj.points[0], y)
