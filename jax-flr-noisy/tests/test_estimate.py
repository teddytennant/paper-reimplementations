import jax
import jax.numpy as jnp
import numpy as np

from flr_noisy.basis import KAPPA, dimension, penalty
from flr_noisy.estimate import fit_fixed_m, fit_flr, gram_and_rhs
from flr_noisy.reconstruct import fourier_coefficients
from flr_noisy.simulate import generate_synthetic


def test_penalty_formula_hand_computed():
    m, n, sigma2 = 3, 50, 0.1
    d = dimension(m)
    assert d == 7
    expected = KAPPA * 7 * 0.1 / 50
    assert np.isclose(penalty(m, n, sigma2), expected)
    assert np.isclose(expected, 0.05614)


def test_criterion_hand_computed():
    gamma = 0.2
    m, n, sigma2 = 3, 50, 0.1
    crit = gamma + penalty(m, n, sigma2)
    assert np.isclose(crit, 0.2 + 4.01 * 7 * 0.1 / 50)


def test_tiny_noise_recovers_in_model_coefficients():
    key = jax.random.PRNGKey(2)
    n, p, m = 80, 48, 2
    d = dimension(m)
    data = generate_synthetic(
        key,
        n,
        p,
        n_kl=21,
        eigenvalue_decay=2.0,
        beta="fourier",
        beta_decay=0.0,
        beta_dim=d,
        sigma2=1e-10,
        tau2=1e-10,
    )
    true = np.zeros(d, dtype=np.float64)
    true[:] = data.beta_coeffs[:d]
    xtilde = fourier_coefficients(data.Z, d)
    alpha, gamma, lam_min = fit_fixed_m(xtilde, data.y, m)
    alpha = np.asarray(alpha)
    assert np.linalg.norm(alpha - true) < 0.05
    assert np.isfinite(float(gamma))
    assert float(lam_min) > 0.0


def test_gram_symmetric_finite():
    key = jax.random.PRNGKey(3)
    data = generate_synthetic(key, 40, 32, n_kl=15, beta_dim=5, sigma2=0.05, tau2=0.05)
    xtilde = fourier_coefficients(data.Z, 7)
    phi, b = gram_and_rhs(xtilde, data.y, 7)
    phi = np.asarray(phi)
    b = np.asarray(b)
    np.testing.assert_allclose(phi, phi.T, atol=1e-14)
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(b))


def test_fit_all_finite_no_nans():
    key = jax.random.PRNGKey(4)
    data = generate_synthetic(key, 64, 48, n_kl=21, beta_dim=7, sigma2=0.1, tau2=0.1)
    result = fit_flr(data.Z, data.y, recon_dim=11, max_m=5, sigma2=data.sigma2)
    assert result.on_gbar
    assert 1 <= result.m_hat <= 5
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.gammas))
    assert np.all(np.isfinite(result.penalties))
    assert np.all(np.isfinite(result.criteria))
    assert np.all(np.isfinite(result.lam_mins))
    assert not np.any(np.isnan(result.alpha))
    from flr_noisy.estimate import predict

    preds = np.asarray(predict(result.alpha, result.recon_coeffs))
    assert np.all(np.isfinite(preds))
    assert preds.shape == (64,)
