import jax
import jax.numpy as jnp
import numpy as np

from flr_noisy.basis import fourier_basis_matrix
from flr_noisy.reconstruct import fourier_coefficients, reconstruct_on_grid
from flr_noisy.simulate import generate_kl_covariates


def test_noiseless_bandlimited_recovery():
    p = 32
    n_basis = 9
    true = np.array([0.4, -1.2, 0.7, 0.1, -0.3, 0.5, 0.2, -0.05, 0.08], dtype=np.float64)
    phi = np.asarray(fourier_basis_matrix(p, n_basis))
    Z = true @ phi
    hat = np.asarray(fourier_coefficients(Z, n_basis))
    np.testing.assert_allclose(hat, true, atol=1e-12)
    rec = np.asarray(reconstruct_on_grid(hat, p))
    np.testing.assert_allclose(rec, Z, atol=1e-12)


def test_reconstruction_mse_increases_with_noise():
    key = jax.random.PRNGKey(0)
    n, p, n_kl, d = 40, 32, 21, 9
    key, kx = jax.random.split(key)
    X, xi, _ = generate_kl_covariates(kx, n, p, n_kl, decay=4.0)
    true_c = xi[:, :d]

    def coeff_mse(tau2, k):
        eta = jax.random.normal(k, (n, p), dtype=jnp.float64) * np.sqrt(tau2)
        Z = X + eta
        hat = fourier_coefficients(Z, d)
        return float(jnp.mean((hat - true_c) ** 2))

    k1, k2 = jax.random.split(key)
    mse_lo = coeff_mse(0.01, k1)
    mse_hi = coeff_mse(1.0, k2)
    assert mse_hi > mse_lo
    assert np.isfinite(mse_lo) and np.isfinite(mse_hi)


def test_reconstruction_mse_decreases_with_more_grid_points():
    key = jax.random.PRNGKey(1)
    n, n_kl, d = 40, 21, 9
    tau2 = 0.25
    key_xi, key_eta = jax.random.split(key)
    lam_sqrt = jnp.arange(1, n_kl + 1, dtype=jnp.float64) ** (-2.0)
    xi = jax.random.normal(key_xi, (n, n_kl), dtype=jnp.float64) * lam_sqrt
    true_c = xi[:, :d]
    eta_unit = jax.random.normal(key_eta, (n, 64), dtype=jnp.float64)

    def coeff_mse(p):
        phi = fourier_basis_matrix(p, n_kl)
        X = xi @ phi
        Z = X + np.sqrt(tau2) * eta_unit[:, :p]
        hat = fourier_coefficients(Z, d)
        return float(jnp.mean((hat - true_c) ** 2))

    mse_32 = coeff_mse(32)
    mse_64 = coeff_mse(64)
    assert mse_64 <= mse_32 * 1.05
    assert mse_64 < mse_32 or np.isclose(mse_64, mse_32, rtol=0.05)
    assert np.isfinite(mse_32) and np.isfinite(mse_64)
