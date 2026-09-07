"""Checks on the score matching objective and its first-order conditions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sme_betel import models
from sme_betel.scorematch import (
    moment_fn,
    moment_matrix,
    score_integrand,
    sm_estimator,
    sm_objective,
)


def _gaussian_data(key, n, m, chol):
    return jax.random.normal(key, (n, m)) @ chol.T


TRUE_PRECISION = jnp.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])


def test_moment_is_zero_at_the_closed_form_estimator():
    """For a mean-zero Gaussian, Jhat is minimized at Lambda = S^{-1}.

    Jhat(Lambda) = -tr(Lambda) + 0.5 tr(Lambda^2 S), so the first-order
    condition of eq (2.4) reads Lambda S + S Lambda = 2 I. The autodiff moment
    function must average to exactly zero there.
    """
    m = 3
    chol = jnp.linalg.cholesky(jnp.linalg.inv(TRUE_PRECISION))
    x = _gaussian_data(jax.random.key(0), 800, m, chol)
    lam_hat = models.gaussian_precision_closed_form(x)
    theta = models.sym_to_vech(lam_hat)

    g = moment_fn(models.gaussian_precision_log_q(m))
    gbar = moment_matrix(g)(x, theta).mean(axis=0)
    scale = jnp.max(jnp.abs(moment_matrix(g)(x, theta)))
    assert float(jnp.max(jnp.abs(gbar))) < 1e-12 * float(scale)


def test_closed_form_estimator_recovers_the_truth_as_n_grows():
    m = 3
    chol = jnp.linalg.cholesky(jnp.linalg.inv(TRUE_PRECISION))
    errors = []
    for n in (250, 2500, 25000, 250000):
        x = _gaussian_data(jax.random.key(1), n, m, chol)
        lam_hat = models.gaussian_precision_closed_form(x)
        errors.append(float(jnp.max(jnp.abs(lam_hat - TRUE_PRECISION))))
    assert errors == sorted(errors, reverse=True)
    # Root-n rate: a hundredfold increase in n should cut the error by about ten.
    assert errors[0] / errors[2] > 4.0
    assert errors[-1] < 0.02


def test_newton_estimator_finds_the_closed_form_minimizer():
    m = 3
    chol = jnp.linalg.cholesky(jnp.linalg.inv(TRUE_PRECISION))
    x = _gaussian_data(jax.random.key(2), 600, m, chol)
    jhat = sm_objective(models.gaussian_precision_log_q(m))
    theta0 = models.sym_to_vech(jnp.eye(m))
    theta = sm_estimator(jhat, x, theta0)
    target = models.sym_to_vech(models.gaussian_precision_closed_form(x))
    assert float(jnp.max(jnp.abs(theta - target))) < 1e-9


def test_moment_matches_central_differences_on_a_nonlinear_model():
    """g is the theta-gradient of the eq (2.3) integrand, checked term by term."""
    m = 3
    theta = jnp.array([1.3, 0.07, -0.4])
    x = jax.random.normal(jax.random.key(3), (12, m)) * 0.8
    g = moment_fn(models.quartic_log_q)

    def integrand(xi, th):
        return score_integrand(models.quartic_log_q, xi, th)

    h = 1e-5
    for xi in x:
        auto = np.asarray(g(xi, theta))
        fd = np.empty_like(auto)
        for j in range(theta.shape[0]):
            e = jnp.zeros_like(theta).at[j].set(h)
            fd[j] = float((integrand(xi, theta + e) - integrand(xi, theta - e)) / (2 * h))
        assert np.max(np.abs(auto - fd)) < 1e-7 * max(1.0, np.max(np.abs(auto)))


def _gauss_hermite(mu, sigma, nodes=80):
    t, w = np.polynomial.hermite.hermgauss(nodes)
    xs = mu + np.sqrt(2.0) * sigma * t
    ws = w / np.sqrt(np.pi)
    return jnp.asarray(xs), jnp.asarray(ws)


def _gauss_legendre(lo, hi, nodes=600):
    t, w = np.polynomial.legendre.leggauss(nodes)
    xs = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
    ws = 0.5 * (hi - lo) * w
    return jnp.asarray(xs), jnp.asarray(ws)


@pytest.mark.parametrize("data", ["gaussian", "quartic"])
def test_hyvarinen_identity_holds_by_quadrature(data):
    """Equation (2.2) is eq (2.1) after an integration by parts.

    E[tr(hess_x log q) + 0.5||grad_x log q||^2] + 0.5 E[||psi_x||^2] must equal
    0.5 E[||psi - psi_x||^2], the relative Fisher divergence of eq (2.1). Both
    sides are computed by quadrature against a data density whose own score is
    known, so this is an exact numerical check, not a Monte Carlo one.
    """
    theta = jnp.array([1.4, 0.06, -0.25])

    def log_q(xi, th):
        return -0.5 * th[0] * (xi[0] - th[2]) ** 2 - th[1] * xi[0] ** 4

    if data == "gaussian":
        mu, sigma = 0.3, 1.2
        xs, ws = _gauss_hermite(mu, sigma)
        psi_x = -(xs - mu) / sigma**2
    else:
        # p_x propto exp(-x^4/4), so the data score is -x^3.
        xs, ws = _gauss_legendre(-7.0, 7.0)
        dens = jnp.exp(-(xs**4) / 4.0)
        ws = ws * dens / jnp.sum(ws * dens)
        ws = ws / jnp.sum(ws)
        psi_x = -(xs**3)

    integrand = jax.vmap(lambda xi: score_integrand(log_q, jnp.array([xi]), theta))(xs)
    psi = jax.vmap(lambda xi: jax.grad(log_q)(jnp.array([xi]), theta)[0])(xs)

    lhs = jnp.sum(ws * integrand) + 0.5 * jnp.sum(ws * psi_x**2)
    rhs = 0.5 * jnp.sum(ws * (psi - psi_x) ** 2)
    assert float(jnp.abs(lhs - rhs)) < 1e-9 * float(jnp.abs(rhs))


def test_normalizing_constant_never_enters_the_moment_function():
    """log p and log q differ by -log Z(theta), which has no x-derivatives."""
    m = 3
    theta = jnp.array([1.1, 0.05, 0.3])
    x = jax.random.normal(jax.random.key(4), (m,))

    def shifted(xi, th):
        return models.quartic_log_q(xi, th) - jnp.log(th[0] ** 3 + jnp.exp(th[2]))

    a = moment_fn(models.quartic_log_q)(x, theta)
    b = moment_fn(shifted)(x, theta)
    assert float(jnp.max(jnp.abs(a - b))) < 1e-12


def test_logistic_moment_is_odd_around_the_section_4_target():
    """The Section 4 target is theta* = 1 and it is exact, not asymptotic.

    p_x is symmetric about 1 and g(x, theta) depends on x only through x-theta
    as an odd function, so E[g(x, 1)] = 0 identically.
    """
    g = moment_fn(models.logistic_location_log_q)
    theta = jnp.array([1.0])
    for d in (0.05, 0.5, 1.0, 3.0, 9.0):
        left = g(jnp.array([1.0 - d]), theta)
        right = g(jnp.array([1.0 + d]), theta)
        assert float(jnp.max(jnp.abs(left + right))) < 1e-12
