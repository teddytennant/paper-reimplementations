import jax
import numpy as np

from flr_noisy.basis import KAPPA, dimension, penalty
from flr_noisy.estimate import fit_flr, select_model
from flr_noisy.reconstruct import fourier_coefficients
from flr_noisy.simulate import generate_synthetic


def test_selector_returns_m_in_range():
    key = jax.random.PRNGKey(5)
    data = generate_synthetic(key, 64, 48, n_kl=21, beta_dim=5, sigma2=0.1, tau2=0.1)
    result = fit_flr(data.Z, data.y, recon_dim=11, max_m=5, sigma2=0.1)
    assert 1 <= result.m_hat <= 5
    assert result.dim == dimension(result.m_hat)
    assert result.dim == result.alpha.shape[0]


def test_smooth_slope_does_not_always_pick_max_m():
    key = jax.random.PRNGKey(6)
    data = generate_synthetic(
        key,
        80,
        64,
        n_kl=21,
        eigenvalue_decay=2.0,
        beta="one_harmonic",
        sigma2=0.1,
        tau2=0.05,
    )
    result = fit_flr(data.Z, data.y, recon_dim=15, max_m=7, sigma2=0.1)
    assert 1 <= result.m_hat <= 7
    assert result.m_hat < 7


def test_plugin_criterion_matches_formula():
    key = jax.random.PRNGKey(7)
    data = generate_synthetic(key, 50, 32, n_kl=15, beta_dim=3, sigma2=0.1, tau2=0.1)
    xtilde = fourier_coefficients(data.Z, 9)
    result = select_model(xtilde, data.y, max_m=4, sigma2=None, kappa=KAPPA)
    n = 50
    for m in range(1, 5):
        d = dimension(m)
        expected_pen = KAPPA * d * result.gammas[m - 1] / n
        assert np.isclose(result.penalties[m - 1], expected_pen)
        assert np.isclose(result.criteria[m - 1], result.gammas[m - 1] + expected_pen)
    assert result.plugin
    assert np.argmin(result.criteria) + 1 == result.m_hat


def test_known_sigma_penalty_matches_hand_value():
    key = jax.random.PRNGKey(8)
    data = generate_synthetic(key, 50, 32, n_kl=15, beta_dim=3, sigma2=0.1, tau2=0.1)
    xtilde = fourier_coefficients(data.Z, 9)
    result = select_model(xtilde, data.y, max_m=3, sigma2=0.1, kappa=KAPPA)
    for m in range(1, 4):
        assert np.isclose(result.penalties[m - 1], penalty(m, 50, 0.1, kappa=KAPPA))
