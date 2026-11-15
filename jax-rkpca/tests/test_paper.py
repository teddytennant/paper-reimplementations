"""Equation checks against independent numpy reconstructions."""

from __future__ import annotations

import numpy as np

from jax_rkpca.distance import half_squared_distance, loadings
from jax_rkpca.kernel import empirical_kernel_map, linear_kernel, rbf_kernel
from jax_rkpca.robust import psi_dot, rotation_fraction
from jax_rkpca.update import init_state, step


def _np_rbf(x, y, sigma):
    d2 = np.sum((np.asarray(x) - np.asarray(y)) ** 2)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def test_rbf_kernel_map_matches_numpy():
    rng = np.random.default_rng(0)
    D, M = 5, 7
    U = rng.normal(size=(M, D))
    x = rng.normal(size=(D,))
    sigma = 0.8
    h = np.asarray(empirical_kernel_map(x, U, kernel="rbf", sigma=sigma))
    h_np = np.array([_np_rbf(U[i], x, sigma) for i in range(M)])
    np.testing.assert_allclose(h, h_np, rtol=1e-10, atol=1e-12)
    # pairwise helper vs the scalar kernel
    np.testing.assert_allclose(
        float(rbf_kernel(U[0], x, sigma)), float(h_np[0]), rtol=1e-10, atol=1e-12
    )


def test_linear_kernel_map_matches_numpy():
    rng = np.random.default_rng(1)
    D, M = 4, 6
    U = rng.normal(size=(M, D))
    x = rng.normal(size=(D,))
    h = np.asarray(empirical_kernel_map(x, U, kernel="linear"))
    h_np = U @ x
    np.testing.assert_allclose(h, h_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        float(linear_kernel(U[2], x)), float(h_np[2]), rtol=1e-10, atol=1e-12
    )


def test_z_matches_numpy_eq3():
    rng = np.random.default_rng(2)
    M, d = 8, 3
    h = rng.normal(size=(M,))
    mu = rng.normal(size=(M,))
    G, _ = np.linalg.qr(rng.normal(size=(M, d)))
    z = float(half_squared_distance(h, mu, G))
    x = h - mu
    P = G @ G.T
    residual = (np.eye(M) - P) @ x
    z_np = 0.5 * np.dot(residual, residual)
    np.testing.assert_allclose(z, z_np, rtol=1e-10, atol=1e-12)
    # equivalent reconstruction when columns are orthonormal
    z_alt = 0.5 * (np.dot(x, x) - np.dot(G.T @ x, G.T @ x))
    np.testing.assert_allclose(z, z_alt, rtol=1e-10, atol=1e-12)


def test_loadings_shape_and_numpy():
    rng = np.random.default_rng(3)
    M, d = 6, 2
    h = rng.normal(size=(M,))
    mu = rng.normal(size=(M,))
    G, _ = np.linalg.qr(rng.normal(size=(M, d)))
    c = np.asarray(loadings(h, mu, G))
    assert c.shape == (d,)
    np.testing.assert_allclose(c, (h - mu) @ G, rtol=1e-10, atol=1e-12)


def test_psi_dot_decreases_with_z():
    z_near, z_far = 0.1, 10.0
    assert float(psi_dot(z_far)) < float(psi_dot(z_near))
    # log-Cauchy derivative is 1/(c+z)
    np.testing.assert_allclose(float(psi_dot(z_near, c=1.0)), 1.0 / (1.0 + z_near))


def test_outliers_get_smaller_alpha():
    alpha0 = 0.5
    a_near = float(rotation_fraction(0.1, alpha0))
    a_far = float(rotation_fraction(1e6, alpha0))
    assert a_far < a_near
    assert a_far < 1e-5


def test_step_loadings_shape():
    M, d = 5, 2
    state = init_state(M, d)
    h = np.arange(M, dtype=float)
    new_state, c_t, info = step(state, h, eta=0.05, alpha0=0.2)
    assert np.asarray(c_t).shape == (d,)
    assert "z" in info and "weight" in info and "alpha" in info
    G = np.asarray(new_state.Gamma)
    np.testing.assert_allclose(G.T @ G, np.eye(d), atol=1e-5)
