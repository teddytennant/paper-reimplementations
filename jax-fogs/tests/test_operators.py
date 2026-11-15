import jax.numpy as jnp

from jax_fogs import (
    fractional_difference,
    gl_psi,
    green_matrices,
    simulate_folti,
)


def test_psi_zero_one_and_recurrence():
    alphas = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert jnp.allclose(gl_psi(alphas, 0.0), 1.0, atol=1e-12)
    assert jnp.allclose(gl_psi(alphas, 1.0), -alphas, atol=1e-12)
    j = jnp.arange(16)
    psi = gl_psi(alphas[:, None], j[None, :])
    ratio = psi[:, 1:] / psi[:, :-1]
    expect = (j[None, :-1] - alphas[:, None]) / (j[None, :-1] + 1.0)
    assert jnp.allclose(ratio, expect, rtol=1e-10, atol=1e-12)


def test_alpha_one_is_first_difference():
    j = jnp.arange(8)
    psi = gl_psi(1.0, j)
    assert jnp.allclose(psi, jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    x = jnp.array(
        [[0.4, -1.2], [0.1, 0.3], [-0.5, 0.8], [1.1, -0.2], [0.0, 0.6]]
    )
    d1 = fractional_difference(x, 1.0)
    assert jnp.allclose(d1, x[1:] - x[:-1], atol=1e-12)
    d_near = fractional_difference(x, 0.999)
    assert jnp.allclose(d_near, x[1:] - x[:-1], atol=5e-3)


def test_green_g0_and_matches_eq3():
    A = jnp.array([[0.25, 0.10], [-0.05, 0.20]])
    alpha = jnp.array([0.2, 0.4])
    x0 = jnp.array([1.0, -0.7])
    t = 12
    G = green_matrices(A, alpha, t)
    assert G.shape == (t + 1, 2, 2)
    assert jnp.allclose(G[0], jnp.eye(2), atol=1e-14)
    eta = jnp.zeros((t, 2))
    x_eq3 = simulate_folti(A, alpha, x0, eta)
    x_eq4 = jnp.einsum("sij,j->si", G, x0)
    assert jnp.allclose(x_eq3, x_eq4, atol=1e-10)
