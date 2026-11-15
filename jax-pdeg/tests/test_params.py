import math

import numpy as np

from jax_pdeg import params_from_L_mu_tau, pdeg_params, warmup_length


def test_pdeg_params_alias():
    assert pdeg_params is params_from_L_mu_tau


def _check_identities(L, mu, tau):
    p = params_from_L_mu_tau(L, mu, tau)
    mu_y = mu + tau
    L_G = 3.0 * L + tau
    s = math.sqrt(2.0 * L * mu_y)
    assert p.mu_y == mu_y
    assert p.L_G == L_G
    assert p.lam == 2.0 * L
    np.testing.assert_allclose(p.h, 1.0 / (64.0 * L_G), rtol=0, atol=0)
    np.testing.assert_allclose(p.beta, p.h * s / 4096.0, rtol=1e-15, atol=0)
    omag = p.h * (8.0 * L - s) / (16.0 + p.h * s)
    np.testing.assert_allclose(p.one_minus_alpha_gamma, omag, rtol=1e-14, atol=1e-16)
    gamma_sqrt = 4.0 * math.sqrt(2.0 * L / mu_y) - 1.0
    np.testing.assert_allclose(p.gamma, gamma_sqrt, rtol=1e-12, atol=1e-12)
    alpha_formula = 1.0 / (1.0 + p.h * s / 16.0)
    np.testing.assert_allclose(p.alpha, alpha_formula, rtol=1e-15, atol=0)
    assert 0.0 < p.alpha < 1.0
    assert 0.0 < p.beta <= 1.0
    assert p.mu_y > 0.0
    assert L_G <= 4.0 * L + 1e-15 or tau > L


def test_parameter_identities_43():
    for L, mu, tau in [
        (1.0, 1.0, 0.0),
        (2.5, 0.4, 0.1),
        (3.0, 0.0, 0.5),
        (10.0, 2.0, 1.0),
    ]:
        _check_identities(L, mu, tau)


def test_lambda_override():
    p = params_from_L_mu_tau(1.0, 1.0, 0.0, lam=3.0)
    assert p.lam == 3.0
    assert p.h == 1.0 / (64.0 * 3.0)


def test_warmup_length_formula():
    beta = 1e-3
    H, eta = 4.0, 1.0
    q = H / eta
    expected = int(math.ceil(math.log(q) / math.log(1.0 + 128.0 * beta)))
    assert warmup_length(beta, H, eta) == expected
    assert warmup_length(beta, 0.5, 1.0) == 0
