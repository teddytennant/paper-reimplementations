import jax
import numpy as np

from flr_noisy.basis import evaluate_fourier, regular_grid
from flr_noisy.estimate import fit_flr, predict, predict_from_grid
from flr_noisy.reconstruct import fourier_coefficients
from flr_noisy.simulate import generate_synthetic


def test_one_harmonic_inner_product_closed_form():
    # β = φ_2, X = φ_2, ⟨β, X⟩_{L²} = 1
    alpha = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_coeffs = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    val = float(predict(alpha, x_coeffs))
    assert np.isclose(val, 1.0)

    p = 64
    t = np.asarray(regular_grid(p))
    beta_grid = np.asarray(evaluate_fourier(t, alpha))
    x_grid = np.asarray(evaluate_fourier(t, x_coeffs))
    discrete = float(np.mean(beta_grid * x_grid))
    assert np.isclose(discrete, 1.0, atol=1e-12)
    from_grid = float(predict_from_grid(alpha, x_grid, recon_dim=3))
    assert np.isclose(from_grid, 1.0, atol=1e-12)


def test_one_harmonic_sine_and_constant():
    # β = φ_3 = √2 sin(2π t), X = φ_3, inner product 1
    alpha = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    assert np.isclose(float(predict(alpha, x)), 1.0)
    # β = φ_1 = 1, X = φ_1, inner product 1
    a1 = np.array([1.0], dtype=np.float64)
    x1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(float(predict(a1, x1)), 1.0)


def test_batch_predict_and_zero_estimator():
    alpha = np.array([0.5, -1.0], dtype=np.float64)
    x = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float64)
    pred = np.asarray(predict(alpha, x))
    np.testing.assert_allclose(pred, np.array([0.5 - 2.0, 0.0 - 1.0]))
    z = np.asarray(predict(np.zeros(0), x))
    np.testing.assert_allclose(z, np.zeros(2))
    assert np.all(np.isfinite(pred))


def test_prediction_finite_on_synthetic():
    key = jax.random.PRNGKey(9)
    data = generate_synthetic(key, 48, 32, n_kl=15, beta="one_harmonic", sigma2=0.05, tau2=0.05)
    result = fit_flr(data.Z, data.y, recon_dim=9, max_m=4, sigma2=0.05)
    pred = np.asarray(predict(result.alpha, result.recon_coeffs))
    true_c = fourier_coefficients(data.X, result.dim)
    pred_true = np.asarray(predict(result.alpha, true_c))
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(pred_true))
    assert pred.shape == (48,)
