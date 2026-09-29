"""Finer Euler–Maruyama grid reduces error vs the closed-form OU (Thm. B.1 / Eq. 78)."""

import jax
import jax.numpy as jnp

from slisde.scan import sequential_simulate
from slisde.sde import (
    affine_additive_mean,
    affine_additive_var,
    em_moments_1d,
    euler_maruyama_pairs,
    ou_mean,
    sample_increments,
)


def test_em_moment_error_decreases_with_finer_grid():
    kappa, sigma, T, z0 = 1.2, 0.8, 1.0, 0.7
    a, b = -kappa, 0.0
    m_cf = affine_additive_mean(z0, a, b, T)
    v_cf = affine_additive_var(sigma, a, T)

    def err(n):
        m, v = em_moments_1d(z0, a, b, sigma, T / n, n)
        return abs(m - m_cf) + abs(v - v_cf)

    e_coarse, e_fine = err(4), err(32)
    assert e_fine < e_coarse
    assert e_fine < 0.05


def test_path_error_vs_closed_form_mean_monotonic():
    """Zero-noise EM path is the mean ODE Euler step; finer grid vs exact OU mean."""
    kappa, T, z0_val = 0.9, 1.0, 1.0
    n_fine, n_coarse = 32, 8
    dt_f, dt_c = T / n_fine, T / n_coarse
    z0 = jnp.array([[z0_val]])
    A = jnp.array([[-kappa]])
    b = jnp.zeros((1,))
    C = jnp.zeros((1, 1, 1))
    D = jnp.array([[0.0]])
    dW_f = jnp.zeros((1, n_fine, 1))
    dW_c = jnp.zeros((1, n_coarse, 1))
    Ff, gf = euler_maruyama_pairs(A, b, C, D, dW_f, dt_f)
    Fc, gc = euler_maruyama_pairs(A, b, C, D, dW_c, dt_c)
    path_f = sequential_simulate(Ff, gf, z0)
    path_c = sequential_simulate(Fc, gc, z0)
    m_cf = ou_mean(z0_val, kappa, T)
    err_f = jnp.abs(path_f[0, -1, 0] - m_cf)
    err_c = jnp.abs(path_c[0, -1, 0] - m_cf)
    assert err_f < err_c


def test_pathwise_error_vs_closed_ou_kernel_monotonic():
    """Left-point variation of constants is the closed-form integral against simple dW.

    X_t = e^{-κ t} X_0 + σ ∫_0^t e^{-κ(t-s)} dW_s, integrand frozen on each fine interval.
    """
    kappa, sigma, T = 1.0, 0.6, 0.8
    n_fine, n_coarse = 64, 8
    ratio = n_fine // n_coarse
    dt_f = T / n_fine
    dt_c = T / n_coarse
    key = jax.random.PRNGKey(9)
    dW_f = sample_increments(key, 1, n_fine, 1, dt_f)[0, :, 0]
    dW_c = dW_f.reshape(n_coarse, ratio).sum(axis=1)
    x0 = 0.5

    times_f = jnp.linspace(0.0, T, n_fine + 1)[:-1]
    closed_T = jnp.exp(-kappa * T) * x0 + sigma * jnp.sum(
        jnp.exp(-kappa * (T - times_f)) * dW_f
    )

    def em_terminal(dW, dt):
        A = jnp.array([[-kappa]])
        b = jnp.zeros((1,))
        C = jnp.zeros((1, 1, 1))
        D = jnp.array([[sigma]])
        F, g = euler_maruyama_pairs(A, b, C, D, dW[None, :, None], dt)
        path = sequential_simulate(F, g, jnp.array([[x0]]))
        return path[0, -1, 0]

    xf = em_terminal(dW_f, dt_f)
    xc = em_terminal(dW_c, dt_c)
    assert abs(xf - closed_T) < abs(xc - closed_T)
