"""Proposition 2: the conditional law of W, its normalising constant, and its moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import kv

from dstmm import (
    gig_log_moment,
    gig_log_norm,
    gig_logpdf,
    gig_moment,
    posterior_w_params,
    sample_gig,
    sample_inverse_gamma,
)

CASES = [(1.0, 3.0, -2.5), (0.05, 9.0, -4.5), (12.0, 0.7, -1.5), (2.0, 2.0, 3.0)]


def _kernel(psi, chi, lam):
    return lambda w: w ** (lam - 1.0) * np.exp(-0.5 * (psi * w + chi / w))


@pytest.mark.parametrize("psi,chi,lam", CASES)
def test_norm_constant_against_quadrature_and_bessel(psi, chi, lam):
    """The constant in equation (13) is 2 K_lam(sqrt(psi chi)) (chi/psi)^(lam/2).

    Two independent oracles: numerical integration of the kernel, and scipy's
    Bessel function in the closed form.
    """
    numeric = quad(_kernel(psi, chi, lam), 0.0, np.inf, limit=400)[0]
    closed = np.log(2.0 * kv(lam, np.sqrt(psi * chi)) * (chi / psi) ** (lam / 2.0))
    got = float(gig_log_norm(psi, chi, lam))
    assert abs(got - np.log(numeric)) < 1e-9
    assert abs(got - closed) < 1e-12


@pytest.mark.parametrize("psi,chi,lam", CASES)
def test_logpdf_integrates_to_one(psi, chi, lam):
    f = lambda w: float(jnp.exp(gig_logpdf(w, psi, chi, lam)))
    assert abs(quad(f, 0.0, np.inf, limit=300)[0] - 1.0) < 1e-8


def test_conditional_law_matches_importance_weighting_from_the_joint():
    """Proposition 2 says W | (y, S=s) ~ GIG(A_s, nu_s + D_js, -(nu_s+p)/2).

    Draw W from the prior IG(nu/2, nu/2) of equation (1), weight each draw by the
    conditional Gaussian of equation (2) at a fixed y, and compare the weighted
    moments and CDF with the analytic GIG ones from (16) and (22). Nothing in the
    reference path uses dstmm.gig beyond the quantity under test, and the error
    budget is the weighted Monte Carlo standard error.
    """
    p = 3
    mu = np.array([0.3, -0.2, 0.5])
    sigma = np.array([[1.4, 0.3, 0.1], [0.3, 0.9, 0.2], [0.1, 0.2, 1.1]])
    alpha = np.array([0.7, -0.4, 0.9])
    nu = 6.0
    y = np.array([1.9, -1.1, 2.4])

    psi, chi, lam = (
        float(v)
        for v in posterior_w_params(
            jnp.asarray(y), jnp.asarray(mu), jnp.asarray(sigma), jnp.asarray(alpha), nu
        )
    )
    assert lam == pytest.approx(-0.5 * (nu + p))

    n = 150_000
    w = np.asarray(sample_inverse_gamma(jax.random.key(2), 0.5 * nu, 0.5 * nu, shape=(n,)))
    resid = y[None, :] - mu[None, :] - w[:, None] * alpha[None, :]
    prec = np.linalg.inv(sigma)
    quad_form = np.einsum("ni,ij,nj->n", resid, prec, resid)
    log_f = -0.5 * p * np.log(w) - 0.5 * quad_form / w
    weights = np.exp(log_f - log_f.max())
    weights /= weights.sum()
    assert 1.0 / np.sum(weights**2) > 0.5 * n  # the prior is a sane proposal here

    for q in (-1.0, 1.0, 2.0):
        est = float(np.sum(weights * w**q))
        se = float(np.sqrt(np.sum(weights**2 * (w**q - est) ** 2)))
        assert abs(est - float(gig_moment(q, psi, chi, lam))) < 5.0 * se

    est = float(np.sum(weights * np.log(w)))
    se = float(np.sqrt(np.sum(weights**2 * (np.log(w) - est) ** 2)))
    assert abs(est - float(gig_log_moment(psi, chi, lam))) < 5.0 * se

    order = np.argsort(w)
    cdf_is = np.cumsum(weights[order])
    norm = quad(_kernel(psi, chi, lam), 0.0, np.inf, limit=400)[0]
    for t in (0.5, 1.0, 2.0, 4.0):
        analytic = quad(_kernel(psi, chi, lam), 0.0, t, limit=400)[0] / norm
        assert abs(float(np.interp(t, w[order], cdf_is)) - analytic) < 0.01


def test_sampler_matches_the_density():
    """The tangent-hull rejection sampler has to reproduce equation (13) itself."""
    psi, chi, lam = 0.8, 5.0, -3.5
    n = 200_000
    w = np.asarray(sample_gig(jax.random.key(0), psi, chi, lam, shape=(n,)))
    assert np.all(w > 0.0)

    for q in (-1.0, 1.0, 2.0):
        est = w**q
        se = est.std() / np.sqrt(n)
        assert abs(est.mean() - float(gig_moment(q, psi, chi, lam))) < 5.0 * se
    se = np.log(w).std() / np.sqrt(n)
    assert abs(np.log(w).mean() - float(gig_log_moment(psi, chi, lam))) < 5.0 * se

    norm = quad(_kernel(psi, chi, lam), 0.0, np.inf, limit=400)[0]
    probs = np.linspace(0.02, 0.98, 49)
    dev = []
    for pr in probs:
        t = float(np.quantile(w, pr))
        dev.append(abs(quad(_kernel(psi, chi, lam), 0.0, t, limit=400)[0] / norm - pr))
    assert max(dev) < 2.5 / np.sqrt(n)  # a Kolmogorov band at about p = 1e-7


def test_inverse_gamma_limit_of_proposition_2():
    """Equations (17)-(18): at A_s = 0 the conditional collapses to IG((nu+p)/2, (nu+D)/2)
    with E(W^-1) = (nu+p)/(nu+D). Reached as a limit of the GIG expressions."""
    nu, p, d = 6.0, 3.0, 4.7
    chi = nu + d
    lam = -0.5 * (nu + p)
    ref = (nu + p) / (nu + d)
    for psi in (1e-6, 1e-8, 1e-10):
        assert abs(float(gig_moment(-1.0, psi, chi, lam)) - ref) < 20.0 * psi
