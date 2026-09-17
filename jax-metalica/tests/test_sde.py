"""Cosine VP-SDE schedule and Euler–Maruyama mean."""

import jax.numpy as jnp
import numpy as np

from jax_metalica.kernels import forward_mean
from jax_metalica.sde import a_t, beta, nu_t, time_grid


def test_a_t_endpoints():
    a0 = float(a_t(0.0))
    a1 = float(a_t(1.0))
    assert abs(a0 - 1.0) < 1e-3
    assert abs(a1 - 0.0) < 2e-2
    assert float(nu_t(0.0)) < 2e-2
    assert abs(float(nu_t(1.0)) - 1.0) < 2e-2


def test_beta_positive_on_unit_interval():
    ts = jnp.linspace(0.01, 0.99, 32)
    b = beta(ts)
    assert jnp.all(b > 0.0)


def test_forward_em_mean_matches_half_beta():
    x = jnp.asarray(1.3)
    t_lo = jnp.asarray(0.25)
    dt = jnp.asarray(0.05)
    mean = forward_mean(x, t_lo, dt)
    expected = x - 0.5 * beta(t_lo) * x * dt
    np.testing.assert_allclose(mean, expected, atol=1e-6)


def test_time_grid_endpoints():
    t = time_grid(8)
    assert t.shape == (9,)
    np.testing.assert_allclose(float(t[0]), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(t[-1]), 1.0, atol=1e-7)
