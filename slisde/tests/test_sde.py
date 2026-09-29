"""Linear SDE layer matches closed-form 1D OU / affine transitions (Eqs. 2, 6, 77, 78)."""

import jax
import jax.numpy as jnp

from slisde.sde import (
    affine_additive_mean,
    affine_additive_var,
    em_moments_1d,
    euler_maruyama_pairs,
    exponential_pairs,
    ou_mean,
    ou_var,
)


def test_em_transition_1d_ou():
    kappa, sigma, dt = 1.25, 0.7, 0.05
    A = jnp.array([[-kappa]])
    b = jnp.zeros((1,))
    C = jnp.zeros((1, 1, 1))
    D = jnp.array([[sigma]])
    dW = jnp.array([[[0.3], [-0.1], [0.2]]])  # (B=1, N=3, m=1)
    F, g = euler_maruyama_pairs(A, b, C, D, dW, dt)
    assert F.shape == (1, 3, 1, 1)
    assert jnp.allclose(F, 1.0 + (-kappa) * dt)
    assert jnp.allclose(g, sigma * dW)


def test_expm_mean_matches_ou_closed_form():
    kappa, sigma, T, n = 0.8, 1.1, 1.0, 16
    dt = T / n
    A = jnp.array([[-kappa]])
    b = jnp.zeros((1,))
    C = jnp.zeros((1, 1, 1))
    D = jnp.array([[sigma]])
    z0 = jnp.array([[1.3]])
    dW = jnp.zeros((1, n, 1))
    F, g = exponential_pairs(A, b, C, D, dW, dt)
    z = z0
    for k in range(n):
        z = F[:, k] @ z[:, :, None]
        z = z[..., 0] + g[:, k]
    assert jnp.allclose(z[0, 0], ou_mean(1.3, kappa, T), atol=1e-10)
    assert jnp.allclose(g, 0.0)


def test_em_moments_match_ou_closed_form():
    kappa, sigma, T = 1.0, 0.9, 0.5
    z0 = 0.4
    a, b = -kappa, 0.0
    n = 256
    dt = T / n
    m_em, v_em = em_moments_1d(z0, a, b, sigma, dt, n)
    m_cf = affine_additive_mean(z0, a, b, T)
    v_cf = affine_additive_var(sigma, a, T)
    assert jnp.allclose(m_em, m_cf, atol=5e-3)
    assert jnp.allclose(v_em, v_cf, atol=5e-3)
    assert jnp.allclose(m_cf, ou_mean(z0, kappa, T))
    assert jnp.allclose(v_cf, ou_var(sigma, kappa, T))


def test_affine_mean_with_drift():
    a, b, T, z0 = -0.5, 0.3, 1.2, 0.2
    m = affine_additive_mean(z0, a, b, T)
    # dm/dt = a m + b, closed form
    ea = jnp.exp(a * T)
    expected = ea * z0 + (b / a) * (ea - 1.0)
    assert jnp.allclose(m, expected)
