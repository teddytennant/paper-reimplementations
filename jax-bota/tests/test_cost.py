"""Invariants of the branched cost (Eq. 7, Lemma 1, 2-source/2-sink toy)."""

import jax.numpy as jnp
import numpy as np

from jax_bota.cost import (
    atomic_branched_cost,
    edge_transport_cost,
    soft_atomic_cost,
    two_source_two_sink_costs,
)


def test_alpha_one_does_not_reward_coalescence_atomic():
    together = atomic_branched_cost(jnp.array([[1.0, 0.0]]), jnp.array([2.0]), alpha=1.0)
    apart = atomic_branched_cost(jnp.array([[1.0, 0.0], [1.0, 0.0]]), jnp.array([1.0, 1.0]), alpha=1.0)
    np.testing.assert_allclose(float(together), float(apart), rtol=1e-5)


def test_alpha_half_rewards_shared_trunk_atomic():
    together = atomic_branched_cost(jnp.array([[1.0, 0.0]]), jnp.array([2.0]), alpha=0.5)
    apart = atomic_branched_cost(
        jnp.array([[1.0, 0.0], [1.0, 0.0]]), jnp.array([1.0, 1.0]), alpha=0.5
    )
    assert float(together) < float(apart) - 1e-6


def test_hand_topology_alpha_one_independent_not_worse():
    independent, branched = two_source_two_sink_costs(1.0)
    # Linear mass: sharing a trunk does not pay. Extra path length makes
    # the branched graph strictly more expensive.
    assert float(independent) <= float(branched) + 1e-6


def test_hand_topology_alpha_half_branched_strictly_cheaper():
    independent, branched = two_source_two_sink_costs(0.5)
    assert float(branched) < float(independent) - 1e-6


def test_edge_cost_alpha_one_is_mass_times_length():
    lengths = jnp.array([2.0, 3.0])
    masses = jnp.array([0.25, 0.75])
    cost = edge_transport_cost(lengths, masses, alpha=1.0)
    np.testing.assert_allclose(float(cost), 2.0 * 0.25 + 3.0 * 0.75, rtol=1e-6)


def test_soft_cost_coalescence_alpha():
    centers = jnp.array([[0.0, 0.0], [4.0, 0.0]])
    v = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    masses = jnp.array([0.5, 0.5])
    together_x = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    apart_x = jnp.array([[0.0, 0.0], [4.0, 0.0]])
    c_together_1 = soft_atomic_cost(together_x, v, centers, alpha=1.0, sigma=0.2, masses=masses)
    c_apart_1 = soft_atomic_cost(apart_x, v, centers, alpha=1.0, sigma=0.2, masses=masses)
    np.testing.assert_allclose(float(c_together_1), float(c_apart_1), rtol=5e-3)
    c_together_h = soft_atomic_cost(together_x, v, centers, alpha=0.5, sigma=0.2, masses=masses)
    c_apart_h = soft_atomic_cost(apart_x, v, centers, alpha=0.5, sigma=0.2, masses=masses)
    assert float(c_together_h) < float(c_apart_h)


def test_soft_cost_matches_atomic_when_centers_are_particles():
    x = jnp.array([[0.0, 0.0], [3.0, 0.0]])
    v = jnp.array([[0.0, 1.0], [0.0, 2.0]])
    m = jnp.array([0.4, 0.6])
    hard = atomic_branched_cost(v, m, alpha=0.7)
    soft = soft_atomic_cost(x, v, x, alpha=0.7, sigma=1e-4, masses=m)
    np.testing.assert_allclose(float(soft), float(hard), rtol=1e-3)
