"""Batched (B, C) shapes for H=2 and H=3; jitted hot path, no Python B/C loops."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from quake import kalman as kalman_mod
from quake.kalman import predict, step, update
from quake.sampler import correct_loop
from quake.transition import observation_row, shift_register


def test_batched_shapes_h2_h3():
    B, C = 6, 5
    for window in (2, 3):
        coeffs = jnp.linspace(0.9, 0.1, window)
        A, b, G = shift_register(coeffs, 0.0, window)
        mean = jnp.zeros((B, C, window))
        cov = jnp.broadcast_to(jnp.eye(window), (B, C, window, window))
        f_hat = jnp.ones((B, C))
        gamma = jnp.linspace(0.7, 1.1, C)
        mu = jnp.zeros((C,))
        Q = jnp.full((C,), 0.02)
        xi = jnp.zeros((C,))
        R = jnp.full((C,), 0.01)

        m1, P1 = jax.jit(predict)(mean, cov, A, b, G, mu, Q)
        assert m1.shape == (B, C, window)
        assert P1.shape == (B, C, window, window)

        H_row = observation_row(gamma, window)
        m2, P2, K = jax.jit(update)(m1, P1, f_hat, H_row, xi, R)
        assert m2.shape == (B, C, window)
        assert P2.shape == (B, C, window, window)
        assert K.shape == (B, C, window)

        m3, P3, f_corr = jax.jit(step)(mean, cov, f_hat, A, b, G, mu, Q, gamma, xi, R)
        assert m3.shape == (B, C, window)
        assert P3.shape == (B, C, window, window)
        assert f_corr.shape == (B, C)


def test_hot_path_has_no_python_batch_loops():
    src = inspect.getsource(kalman_mod)
    assert "for b in" not in src
    assert "for c in" not in src
    assert "for batch" not in src
    assert "for channel" not in src


def test_correct_loop_batched():
    T, B, C, window = 8, 4, 3, 2
    A, b, G = shift_register(jnp.array([0.75, 0.25]), 0.0, window)
    f_hats = jnp.ones((T, B, C))
    init_mean = jnp.zeros((B, C, window))
    init_cov = jnp.broadcast_to(jnp.eye(window), (B, C, window, window))
    f_corr, state = correct_loop(
        f_hats,
        init_mean,
        init_cov,
        A,
        b,
        G,
        mu=jnp.zeros((C,)),
        Q=jnp.full((C,), 0.03),
        gamma=jnp.ones((C,)),
        xi=jnp.zeros((C,)),
        R=jnp.full((C,), 0.01),
    )
    assert f_corr.shape == (T, B, C)
    assert state.mean.shape == (B, C, window)
    assert state.cov.shape == (B, C, window, window)
    np.testing.assert_array_equal(np.asarray(jnp.isfinite(f_corr)), True)
