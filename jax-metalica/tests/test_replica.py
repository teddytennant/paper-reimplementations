"""Swap acceptance and Appendix E closed-form kernel ratio."""

import jax
import jax.numpy as jnp
import numpy as np

from jax_metalica.bias import cv_grid, deposit_hills
from jax_metalica.gmm import toy_gmm
from jax_metalica.kernels import (
    log_B0_minus_log_B,
    log_normal,
    pretrained_backward_mean,
    tilted_backward_mean,
)
from jax_metalica.replica import attempt_swap, swap_acceptance
from jax_metalica.sde import beta, time_grid


def test_swap_A_in_unit_interval():
    key = jax.random.PRNGKey(0)
    gmm = toy_gmm(0.0)
    grid = cv_grid(41)
    times = time_grid(8)
    t_lo, t_hi = times[2], times[3]
    k1, k2, k3 = jax.random.split(key, 3)
    x_lo = jax.random.normal(k1)
    x_hi = jax.random.normal(k2)
    _new_lo, _new_hi, A = attempt_swap(
        k3, x_lo, x_hi, t_lo, t_hi, grid.omega, grid.V, gmm, rho=0.5
    )
    A = float(A)
    assert 0.0 <= A <= 1.0

    y_lo = jnp.asarray(0.4)
    y_hi = jnp.asarray(-0.2)
    A2 = float(
        swap_acceptance(
            x_lo, x_hi, y_lo, y_hi, t_lo, t_hi, grid.omega, grid.V, gmm, rho=0.5
        )
    )
    assert 0.0 <= A2 <= 1.0


def test_eq21_closed_form_matches_two_gaussians():
    key = jax.random.PRNGKey(1)
    gmm = toy_gmm(0.0)
    grid = cv_grid(81)
    # Nonzero bias so ∇r is not identically zero.
    V, _ = deposit_hills(
        grid.omega, grid.V, jnp.array([3.0, 2.5]), h=0.4, sigma=0.4, gamma=2.0
    )
    t_hi = jnp.asarray(0.4)
    dt = jnp.asarray(0.05)
    k1, k2 = jax.random.split(key)
    y = jax.random.normal(k1) * 0.5
    x = jax.random.normal(k2) * 0.5
    closed = log_B0_minus_log_B(x, y, t_hi, dt, grid.omega, V, gmm, rho=0.5)
    mu_p = pretrained_backward_mean(y, t_hi, dt, gmm)
    mu_q = tilted_backward_mean(y, t_hi, dt, grid.omega, V, gmm, rho=0.5)
    v = beta(t_hi) * dt
    explicit = log_normal(x, mu_p, v) - log_normal(x, mu_q, v)
    np.testing.assert_allclose(closed, explicit, atol=1e-5, rtol=1e-5)
