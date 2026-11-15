"""Slice helpers and walk invariant: every state has exactly k plus-ones."""

import jax
import jax.numpy as jnp
import numpy as np

from himag.slice import (
    enumerate_slice,
    index_set_from_spins,
    magnetization,
    n_plus,
    random_slice_spin,
    spins_from_index_set,
)
from himag.sk import sample_goe
from himag.walk import down_up_step


def test_enumerate_counts_and_k():
    d, k = 6, 2
    xs = np.asarray(enumerate_slice(d, k))
    assert xs.shape == (15, 6)
    assert np.all(xs * xs == 1.0)
    assert np.all(n_plus(xs) == k)
    assert np.all(magnetization(xs) == -d + 2 * k)


def test_index_set_roundtrip():
    d, k = 8, 3
    x = spins_from_index_set([0, 3, 7], d)
    assert list(index_set_from_spins(x)) == [0, 3, 7]
    assert int(n_plus(x)) == k


def test_walk_preserves_slice():
    d, k, beta = 8, 3, 0.7
    key = jax.random.PRNGKey(0)
    kJ, kx, kw = jax.random.split(key, 3)
    J = sample_goe(kJ, d)
    h = jax.random.normal(kx, (d,))
    x = random_slice_spin(kw, d, k)
    assert int(n_plus(x)) == k
    keys = jax.random.split(jax.random.PRNGKey(1), 40)
    for sub in keys:
        x = down_up_step(sub, x, J, h, beta)
        assert int(n_plus(x)) == k
        assert int(jnp.sum(jnp.abs(x) == 1.0)) == d
        assert bool(jnp.all(jnp.isfinite(x)))
