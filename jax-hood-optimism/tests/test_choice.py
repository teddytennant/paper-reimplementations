"""Lifted entropic choice map of eqs. (19)-(21)."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from hood.choice import (
    entropic_choice,
    lifted_choice,
    lifted_value,
    regularizer_constant,
    solve_lambda,
    temperature,
)


def test_lifted_choice_on_simplex():
    key = jax.random.key(0)
    for k in (2, 5, 8):
        key, sub = jax.random.split(key)
        y = jax.random.normal(sub, (k,))
        q = np.asarray(lifted_choice(y, regularizer_constant(k)))
        assert np.all(q >= -1e-12)
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-10)


def test_solve_lambda_matches_dense_grid():
    key = jax.random.key(1)
    grid = jnp.linspace(1e-6, 1.0 - 1e-6, 20000)
    for k in (2, 4, 7):
        L = regularizer_constant(k)
        for _ in range(4):
            key, sub = jax.random.split(key)
            y = jax.random.normal(sub, (k,))
            lam = float(solve_lambda(y, L))
            vals = jax.vmap(lambda t: lifted_value(t, y, L))(grid)
            grid_lam = float(grid[int(jnp.argmax(vals))])
            assert abs(lam - grid_lam) < 1e-4


def test_temperature_endpoints():
    np.testing.assert_allclose(float(temperature(jnp.float64(0.0))), 0.0)
    np.testing.assert_allclose(float(temperature(jnp.float64(1.0))), 0.5)
    y = jnp.array([1.0, 2.0, 3.0])
    q = jax.nn.softmax(temperature(jnp.float64(0.0)) * y)
    np.testing.assert_allclose(np.asarray(q), np.full(3, 1.0 / 3.0))


def test_regularizer_constant():
    for k in (2, 3, 10):
        log_k = math.log(k)
        expected = 4.0 + log_k + 0.25 * log_k * log_k
        np.testing.assert_allclose(regularizer_constant(k), expected)


def test_entropic_choice_is_softmax():
    y = jnp.array([0.2, -1.0, 3.5])
    np.testing.assert_allclose(np.asarray(entropic_choice(y)), np.asarray(jax.nn.softmax(y)))
