"""Appendix A: the derived bias and variance against a Monte Carlo estimate.

Everything here runs in float64 because the quantities under test are order
1e-3 and the appendix's reported gap between the two variances is 1.5e-3.
"""

import numpy as np
import pytest

from mfspnet.estimators import ema_weights
from mfspnet.signal_noise import (
    analytic_mean,
    analytic_moments,
    analytic_variance,
    monte_carlo_moments,
    paper_scenario,
    variance_gap,
    SignalNoiseModel,
)

MODEL = paper_scenario()


def test_everything_is_float64():
    for arr in (MODEL.mu_val, MODEL.mu_train, MODEL.sigma_val, MODEL.sigma_train):
        assert arr.dtype == np.float64


def test_analytic_mean_matches_monte_carlo():
    """Eqs. 15 and 18 against the sample mean of 200k replications."""
    mc = monte_carlo_moments(MODEL, n_replications=200_000, seed=11)
    exact = analytic_moments(MODEL)
    # standard error of the mean at M = 2e5 with Var ~ 3e-3 is about 1.2e-4
    assert mc.bias_vle == pytest.approx(exact.bias_vle, abs=5e-4)
    assert mc.bias_tse == pytest.approx(exact.bias_tse, abs=5e-4)


def test_analytic_variance_matches_monte_carlo():
    """Eqs. 16 and 19 against the sample variance of 200k replications."""
    mc = monte_carlo_moments(MODEL, n_replications=200_000, seed=12)
    exact = analytic_moments(MODEL)
    assert mc.var_vle == pytest.approx(exact.var_vle, rel=0.02)
    assert mc.var_tse == pytest.approx(exact.var_tse, rel=0.02)


def test_variance_gap_equation_20():
    """Eq. 20 must equal the difference of Eqs. 19 and 16, and be positive
    here, which is the "VLE-EMA is the calmer estimator" claim."""
    exact = analytic_moments(MODEL)
    assert variance_gap(MODEL) == pytest.approx(exact.var_tse - exact.var_vle, rel=1e-12)
    assert variance_gap(MODEL) > 0.0


def test_net_effect_goes_the_way_the_appendix_says():
    """Both halves of "Net effect": lower bias and lower variance for VLE-EMA."""
    for moments in (analytic_moments(MODEL), monte_carlo_moments(MODEL, 200_000, 13)):
        assert abs(moments.bias_vle) < abs(moments.bias_tse)
        assert moments.var_vle < moments.var_tse


def test_reproduces_the_reported_monte_carlo_numbers():
    """The appendix reports bias about -0.01 vs -0.045 and variance 0.0026 vs
    0.0041. These are the analytic values of the reconstructed scenario."""
    m = analytic_moments(MODEL)
    assert m.bias_vle == pytest.approx(-0.01, abs=2e-3)
    assert m.bias_tse == pytest.approx(-0.045, abs=2e-3)
    assert m.var_vle == pytest.approx(0.0026, abs=1e-4)
    assert m.var_tse == pytest.approx(0.0041, abs=1e-4)


def test_appendix_scale_monte_carlo_is_consistent_within_its_own_error():
    """The appendix ran M = 1000. At that size the standard error on the bias
    is sqrt(Var/M) ~ 1.6e-3, so agreement is only ever going to be loose."""
    exact = analytic_moments(MODEL)
    for seed in range(8):
        mc = monte_carlo_moments(MODEL, n_replications=1000, seed=seed)
        se_vle = np.sqrt(exact.var_vle / 1000)
        se_tse = np.sqrt(exact.var_tse / 1000)
        assert abs(mc.bias_vle - exact.bias_vle) < 4 * se_vle
        assert abs(mc.bias_tse - exact.bias_tse) < 4 * se_tse
        assert abs(mc.bias_vle) < abs(mc.bias_tse)
        assert mc.var_vle < mc.var_tse


def test_bias_is_driven_by_the_train_val_gap_not_the_weights():
    """The appendix's bias argument is that mu_train sits under the final
    validation loss. Close that gap and the TSE bias advantage disappears."""
    m = SignalNoiseModel(
        mu_val=MODEL.mu_val,
        mu_train=MODEL.mu_val.copy(),
        sigma_val=MODEL.sigma_val,
        sigma_train=MODEL.sigma_train,
    )
    exact = analytic_moments(m)
    assert exact.bias_vle == pytest.approx(exact.bias_tse)
    assert exact.var_vle < exact.var_tse  # variance argument survives on its own


def test_variance_formula_reduces_to_the_iid_case():
    """With a flat sigma, Eq. 16 must equal sigma^2 * sum(w^2) / (sum w)^2."""
    T, gamma, sigma = 15, 0.78, 0.3
    w = ema_weights(T, gamma)
    expected = sigma**2 * (w**2).sum() / w.sum() ** 2
    assert analytic_variance(np.full(T, sigma), gamma) == pytest.approx(expected)


def test_mean_formula_reduces_to_a_constant_trend():
    assert analytic_mean(np.full(15, 0.7), 0.78) == pytest.approx(0.7)


def test_lower_gamma_concentrates_weight_and_raises_variance():
    """Fewer effective epochs means a noisier estimate."""
    sigma = np.full(15, 0.25)
    assert analytic_variance(sigma, 0.5) > analytic_variance(sigma, 0.9)
