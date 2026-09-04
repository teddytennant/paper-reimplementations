"""Eq. 1 NTK, the kernel solve of Eq. 2, and closed-form regression of Eqs. 4-5."""

import jax
import jax.numpy as jnp
import numpy as np

from metaquill_kip.field import MLPConfig, encode, encoded_dim, init_params, make_field
from metaquill_kip.ntk import (
    jacobian_matrix,
    kernel_predict,
    kernel_solve,
    ntk_matrix,
    ntk_matrix_jvp,
    ntk_regression,
)


def _tiny_field(seed=0, width=8, depth=2, k_bands=2):
    cfg = MLPConfig(in_dim=encoded_dim(2, "fourier", k_bands), width=width, depth=depth)
    params = init_params(cfg, jax.random.PRNGKey(seed))
    return make_field(cfg), params, lambda x: encode(x, "fourier", k_bands)


def test_ntk_matrix_is_symmetric_psd():
    f, params, enc = _tiny_field()
    Z = enc(jax.random.uniform(jax.random.PRNGKey(1), (12, 2), minval=-1.0, maxval=1.0))
    K = ntk_matrix(f, params, Z)
    assert jnp.max(jnp.abs(K - K.T)) < 1e-12
    eigs = jnp.linalg.eigvalsh(0.5 * (K + K.T))
    assert float(jnp.min(eigs)) >= -1e-8


def test_ntk_matrix_matches_jvp():
    f, params, enc = _tiny_field()
    Z1 = enc(jax.random.uniform(jax.random.PRNGKey(2), (7, 2), minval=-1.0, maxval=1.0))
    Z2 = enc(jax.random.uniform(jax.random.PRNGKey(3), (5, 2), minval=-1.0, maxval=1.0))
    K = ntk_matrix(f, params, Z1, Z2)
    Kj = ntk_matrix_jvp(f, params, Z1, Z2)
    np.testing.assert_allclose(np.asarray(K), np.asarray(Kj), atol=1e-8)
    Ks = ntk_matrix(f, params, Z1)
    Kjs = ntk_matrix_jvp(f, params, Z1)
    np.testing.assert_allclose(np.asarray(Ks), np.asarray(Kjs), atol=1e-8)


def test_kernel_solve_interpolates():
    f, params, enc = _tiny_field()
    Z = enc(jax.random.uniform(jax.random.PRNGKey(4), (8, 2), minval=-1.0, maxval=1.0))
    Y = jax.random.normal(jax.random.PRNGKey(5), (8,), dtype=jnp.float64)
    K = ntk_matrix(f, params, Z)
    W = kernel_solve(K, Y, ridge=1e-10, method="cholesky")
    np.testing.assert_allclose(np.asarray(K @ W), np.asarray(Y), atol=1e-6, rtol=1e-6)
    W0 = kernel_solve(K, Y, ridge=0.0, method="lstsq")
    np.testing.assert_allclose(np.asarray(kernel_predict(K, W0)), np.asarray(Y), atol=1e-5, rtol=1e-5)


def test_ntk_regression_reconstructs_train_points():
    f, params, enc = _tiny_field(width=16, depth=2, k_bands=3)
    coords = jax.random.uniform(jax.random.PRNGKey(6), (16, 2), minval=-1.0, maxval=1.0)
    Z = enc(coords)
    Y = jax.random.normal(jax.random.PRNGKey(7), (16,), dtype=jnp.float64) * 0.3
    pred, _ = ntk_regression(f, params, Z, Y, Z, ridge=1e-8)
    np.testing.assert_allclose(np.asarray(pred), np.asarray(Y), atol=1e-5, rtol=1e-5)


def test_jacobian_gram_matches_ntk_matrix():
    f, params, enc = _tiny_field()
    Z = enc(jax.random.uniform(jax.random.PRNGKey(8), (9, 2), minval=-1.0, maxval=1.0))
    J = jacobian_matrix(f, params, Z)
    K = ntk_matrix(f, params, Z)
    np.testing.assert_allclose(np.asarray(J @ J.T), np.asarray(K), atol=1e-10)
