"""Exact compositional score for a 2D linear-Gaussian model.

This is the task's checkable Gaussian, not the 10-dimensional correlated
Gaussian of Appendix A.5. The A.5 lesson that is kept: the sampler is driven by
the exact score of the VP-noised posterior, and its mean is compared to the
analytic Gaussian posterior mean.

Model
    theta ~ N(0, I_2)
    y | theta, xi ~ N(A(xi) theta, sigma^2 I_2)
    sigma = 0.1, A(xi) = [[1, 0], [xi, 1]]
    n = 8 designs xi_i = 2 i / (n - 1) - 1, theta_true = [1.0, -0.7]

Score target
    The compositional posterior score is the clean identity

        s_post(theta) = s_prior(theta) + sum_i s(y_i | theta, xi_i)

    which equals the score of the closed-form posterior (prior times the product
    of likelihoods). Summing per-observation likelihood scores needs no Jacobian
    correction; that correction appears only when one instead sums diffusion-time
    scores of separately noised single-observation posteriors (the JAC target).

    The reverse SDE is then driven by the score of the noised joint posterior,
    the F-NPSE-SDE target of Appendix A.5:

        theta_t | data ~ N(alpha(t) m, alpha(t)^2 C + V_n(t) I)
        s_t(theta_t) = -C_t^{-1} (theta_t - alpha(t) m)

    where (m, C) is the compositional Gaussian posterior and V_n is equation (26).
    Building (m, C) by summing likelihood precisions, then noising, matches the
    score of that same noised Gaussian at every noise level.
"""

import jax
import jax.numpy as jnp

from hbds.diffusion import (
    marginal_variance,
    reverse_sde_sample,
    signal_scale,
)

SIGMA = 0.1
SIGMA2 = SIGMA ** 2
THETA_TRUE = jnp.array([1.0, -0.7])
N_OBS = 8


def design_matrix(xi):
    """A(xi) = [[1, 0], [xi, 1]]."""
    xi = jnp.asarray(xi, dtype=jnp.float64)
    one = jnp.ones_like(xi)
    zero = jnp.zeros_like(xi)
    return jnp.stack(
        [jnp.stack([one, zero]), jnp.stack([xi, one])],
        axis=0,
    )


def designs(n=N_OBS):
    return 2.0 * jnp.arange(n, dtype=jnp.float64) / (n - 1) - 1.0


def simulate_observations(key, n=N_OBS, theta_true=THETA_TRUE, sigma=SIGMA):
    """Draw y_i = A(xi_i) theta_true + sigma * eps_i."""
    theta_true = jnp.asarray(theta_true, dtype=jnp.float64)
    xis = designs(n)
    noise = sigma * jax.random.normal(key, (n, 2), dtype=jnp.float64)

    def one(xi, eps):
        return design_matrix(xi) @ theta_true + eps

    ys = jax.vmap(one)(xis, noise)
    return xis, ys, theta_true


def prior_score(theta):
    """Score of N(0, I)."""
    return -theta


def likelihood_score(theta, y, xi, sigma2=SIGMA2):
    """Exact score of one observation: grad_theta log N(y; A(xi) theta, sigma^2 I)."""
    A = design_matrix(xi)
    resid = y - A @ theta
    return (A.T @ resid) / sigma2


def compositional_score(theta, ys, xis, sigma2=SIGMA2):
    """s_prior + sum_i s(y_i | theta, xi_i). Clean posterior score."""
    lik = jax.vmap(lambda y, xi: likelihood_score(theta, y, xi, sigma2))(ys, xis)
    return prior_score(theta) + jnp.sum(lik, axis=0)


def single_observation_posterior_score(theta, y, xi, sigma2=SIGMA2):
    """Score of p(theta | y, xi) for one observation."""
    return prior_score(theta) + likelihood_score(theta, y, xi, sigma2)


def factorized_posterior_score(theta, ys, xis, sigma2=SIGMA2):
    """sum_i s(theta | y_i) - (n - 1) s_prior.

    This is the product identity p(theta | D) proportional to
    p(theta)^{1-n} * product_i p(theta | y_i). It equals the compositional
    likelihood sum, with no score Jacobian.
    """
    n = ys.shape[0]
    parts = jax.vmap(
        lambda y, xi: single_observation_posterior_score(theta, y, xi, sigma2)
    )(ys, xis)
    return jnp.sum(parts, axis=0) - (n - 1) * prior_score(theta)


def _precision_sum(ys, xis, sigma2=SIGMA2):
    """Joint precision and linear term by summing per-observation likelihoods."""

    def one(y, xi):
        A = design_matrix(xi)
        prec = (A.T @ A) / sigma2
        lin = (A.T @ y) / sigma2
        return prec, lin

    precs, lins = jax.vmap(one)(ys, xis)
    lam = jnp.eye(2, dtype=jnp.float64) + jnp.sum(precs, axis=0)
    lin = jnp.sum(lins, axis=0)
    return lam, lin


def _precision_from_factors(ys, xis, sigma2=SIGMA2):
    """Same joint precision via (1-n) prior + sum of single-observation posteriors."""

    def one(y, xi):
        A = design_matrix(xi)
        lam_i = jnp.eye(2, dtype=jnp.float64) + (A.T @ A) / sigma2
        lin_i = (A.T @ y) / sigma2
        return lam_i, lin_i

    lams, lins = jax.vmap(one)(ys, xis)
    n = ys.shape[0]
    lam = (1 - n) * jnp.eye(2, dtype=jnp.float64) + jnp.sum(lams, axis=0)
    lin = jnp.sum(lins, axis=0)
    return lam, lin


def _mean_cov(lam, lin):
    cov = jnp.linalg.inv(lam)
    mean = cov @ lin
    return mean, cov


def analytic_posterior(ys, xis, sigma2=SIGMA2):
    """Closed-form posterior N(m, C) from prior times the product of likelihoods."""
    return _mean_cov(*_precision_sum(ys, xis, sigma2))


def analytic_posterior_from_factors(ys, xis, sigma2=SIGMA2):
    """Same posterior, built from the per-observation posterior factorization."""
    return _mean_cov(*_precision_from_factors(ys, xis, sigma2))


def log_posterior_density(theta, ys, xis, sigma2=SIGMA2):
    """Log of prior times product of likelihoods, up to a theta-free constant."""
    lp = -0.5 * jnp.dot(theta, theta)

    def one(y, xi):
        A = design_matrix(xi)
        resid = y - A @ theta
        return -0.5 * jnp.dot(resid, resid) / sigma2

    return lp + jnp.sum(jax.vmap(one)(ys, xis))


def noised_gaussian_score(theta_t, t, mean, cov, n):
    """Score of N(alpha(t) m, alpha(t)^2 C + V_n(t) I). Appendix A.5 target."""
    a = signal_scale(t)
    v = marginal_variance(t, n)
    d = cov.shape[-1]
    ct = (a ** 2) * cov + v * jnp.eye(d, dtype=jnp.float64)
    centered = theta_t - a * mean
    if theta_t.ndim == 1:
        return -jnp.linalg.solve(ct, centered)
    return -jnp.linalg.solve(ct, centered.T).T


def compositional_noised_score(theta_t, t, ys, xis, sigma2=SIGMA2):
    """Noised joint score from the compositional (precision-sum) posterior."""
    mean, cov = analytic_posterior(ys, xis, sigma2)
    return noised_gaussian_score(theta_t, t, mean, cov, ys.shape[0])


def factorized_noised_score(theta_t, t, ys, xis, sigma2=SIGMA2):
    """Noised joint score from the per-observation posterior factorization."""
    mean, cov = analytic_posterior_from_factors(ys, xis, sigma2)
    return noised_gaussian_score(theta_t, t, mean, cov, ys.shape[0])


def log_noised_density(theta_t, t, mean, cov, n):
    """Log density of the VP-noised posterior, up to a theta-free constant."""
    a = signal_scale(t)
    v = marginal_variance(t, n)
    d = cov.shape[-1]
    ct = (a ** 2) * cov + v * jnp.eye(d, dtype=jnp.float64)
    resid = theta_t - a * mean
    solved = jnp.linalg.solve(ct, resid)
    return -0.5 * jnp.dot(resid, solved)


def terminal_sample(key, n_samples, mean, cov, n, t=1.0):
    """Draw from the prescribed terminal law N(alpha(t) m, alpha(t)^2 C + V_n(t) I)."""
    a = signal_scale(t)
    v = marginal_variance(t, n)
    d = cov.shape[-1]
    ct = (a ** 2) * cov + v * jnp.eye(d, dtype=jnp.float64)
    chol = jnp.linalg.cholesky(ct)
    z = jax.random.normal(key, (n_samples, d), dtype=jnp.float64)
    return a * mean + z @ chol.T


def sample_posterior(ys, xis, n_samples, n_steps, key, t_end=1e-3, sigma2=SIGMA2):
    """Reverse SDE with the exact compositional noised score.

    Returns samples, the analytic posterior mean, and the terminal initialization
    (so a caller can check that the chain actually moved).
    """
    mean, cov = analytic_posterior(ys, xis, sigma2)
    n = int(ys.shape[0])
    key, k_init = jax.random.split(key)
    x0 = terminal_sample(k_init, n_samples, mean, cov, n, t=1.0)

    def score_fn(state, t):
        return noised_gaussian_score(state, t, mean, cov, n)

    samples = reverse_sde_sample(x0, score_fn, n, n_steps, key, t_start=1.0, t_end=t_end)
    return samples, mean, x0
