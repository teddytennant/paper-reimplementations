"""Checks on the SME-BETEL posterior of eq (2.7) and the Algorithm S1 sampler."""

import jax
import jax.numpy as jnp
import numpy as np

from sme_betel import models
from sme_betel.asymptotics import gamma_matrix, sandwich_covariance
from sme_betel.posterior import log_posterior_fn, metropolis_hastings
from sme_betel.scorematch import moment_fn, moment_matrix

M = 2


def _heavy_tailed(key, n, df=8.0):
    """Elliptical t data, so the Gaussian working model is misspecified."""
    z = jax.random.normal(key, (n, M))
    u = jax.random.chisquare(jax.random.fold_in(key, 1), df, (n, 1))
    return z * jnp.sqrt(df / u) / jnp.sqrt(df / (df - 2.0))


def test_metropolis_hastings_targets_the_right_distribution():
    """Algorithm S1 on a Gaussian target, where the answer is known."""
    mean = jnp.array([0.7, -1.2])
    cov = jnp.array([[1.0, 0.6], [0.6, 2.0]])
    prec = jnp.linalg.inv(cov)

    def log_target(theta):
        d = theta - mean
        return -0.5 * d @ prec @ d

    out = metropolis_hastings(
        log_target,
        jnp.zeros(2),
        jax.random.key(0),
        n_iter=60_000,
        burn_in=10_000,
        step_chol=jnp.linalg.cholesky(cov),
    )
    draws = np.asarray(out.samples)
    assert 0.25 < float(out.accept_rate) < 0.6
    assert np.max(np.abs(draws.mean(0) - np.asarray(mean))) < 0.05
    assert np.max(np.abs(np.cov(draws.T) - np.asarray(cov))) < 0.15


def test_posterior_is_minus_infinity_outside_the_convex_hull():
    """The discard rule stated after eq (2.7) and used in Algorithm S1.

    For the mean-zero Gaussian precision model, g(x, Lambda) = -I + Lambda x x'
    contracted onto the free entries. Push Lambda far enough down and every
    g(x_i, Lambda) has a negative diagonal, so the origin leaves the hull.
    """
    x = _heavy_tailed(jax.random.key(1), 400)
    gmat = moment_matrix(moment_fn(models.gaussian_precision_log_q(M)))
    log_post = log_posterior_fn(gmat, lambda t: 0.0)
    good = models.sym_to_vech(models.gaussian_precision_closed_form(x))
    assert np.isfinite(float(log_post(good, x)))
    assert float(log_post(jnp.array([1e-6, 0.0, 1e-6]), x)) == -np.inf
    assert float(log_post(jnp.array([-1.0, 0.0, -1.0]), x)) == -np.inf


def test_sampler_never_leaves_the_prior_support():
    x = _heavy_tailed(jax.random.key(2), 300)
    gmat = moment_matrix(moment_fn(models.gaussian_precision_log_q(M)))
    lo, hi = 0.6, 1.6

    def log_prior(theta):
        inside = jnp.all((theta[jnp.array([0, 2])] > lo) & (theta[jnp.array([0, 2])] < hi))
        return jnp.where(inside, 0.0, -jnp.inf)

    log_post = log_posterior_fn(gmat, log_prior)
    start = models.sym_to_vech(models.gaussian_precision_closed_form(x))
    out = metropolis_hastings(
        lambda t: log_post(t, x), start, jax.random.key(3), 4000, 1000, init_scale=0.05
    )
    diag = np.asarray(out.samples)[:, [0, 2]]
    assert diag.min() > lo and diag.max() < hi


def test_posterior_covariance_matches_the_sandwich_of_theorem_1():
    """Theorem 1, eq (4.2): the posterior covariance is n^{-1}(Gamma' Delta^{-1} Gamma)^{-1}.

    Working model is a mean-zero Gaussian with unknown precision, the data are
    heavy tailed, so this is the misspecified case where the sandwich and the
    model-based curvature disagree.
    """
    n = 1000
    x = _heavy_tailed(jax.random.key(4), n)
    log_q = models.gaussian_precision_log_q(M)
    g = moment_fn(log_q)
    gmat = moment_matrix(g)

    theta_hat = models.sym_to_vech(models.gaussian_precision_closed_form(x))
    target = sandwich_covariance(g, x, theta_hat) / n

    log_post = log_posterior_fn(gmat, lambda t: -0.5 * jnp.sum(t**2) / 100.0)
    out = jax.jit(
        lambda k: metropolis_hastings(
            lambda t: log_post(t, x),
            theta_hat,
            k,
            n_iter=20_000,
            burn_in=5_000,
            step_chol=jnp.linalg.cholesky(target),
            init_scale=1.0,
        )
    )(jax.random.key(5))
    draws = np.asarray(out.samples)
    post_cov = np.cov(draws.T)
    tgt = np.asarray(target)

    assert 0.2 < float(out.accept_rate) < 0.6
    # Posterior mean sits on theta_hat_n to within a fraction of a posterior sd.
    sd = np.sqrt(np.diag(tgt))
    assert np.max(np.abs(draws.mean(0) - np.asarray(theta_hat)) / sd) < 0.4

    ratio = np.diag(post_cov) / np.diag(tgt)
    assert np.max(np.abs(ratio - 1.0)) < 0.3
    rel = np.linalg.norm(post_cov - tgt) / np.linalg.norm(tgt)
    assert rel < 0.3

    # The sandwich is the right target and the model-based curvature Gamma^{-1}/n
    # is not. Under this much misspecification the two differ by a factor of
    # roughly three, so the comparison is not a coin flip across seeds.
    naive = np.linalg.inv(np.asarray(gamma_matrix(g, x, theta_hat))) / n
    rel_naive = np.linalg.norm(post_cov - naive) / np.linalg.norm(tgt)
    assert rel_naive > 0.45
    assert rel_naive > 2.0 * rel
