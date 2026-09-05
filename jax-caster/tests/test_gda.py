"""Equation (5), the Gaussian discriminant."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import multivariate_normal  # noqa: E402

from caster.gda import confidence, discriminant_scores, predict  # noqa: E402
from caster.synthetic import random_spd  # noqa: E402


def test_scores_match_the_gaussian_log_density_up_to_the_dropped_constant():
    """Eq. (5) is log(pi_c N(z; mu_c, Sigma_c)) with the k/2 log 2pi term dropped."""
    key = jax.random.PRNGKey(0)
    k, c, n = 5, 4, 30
    key_z, key_mu, key_cov, key_pi = jax.random.split(key, 4)
    z = jax.random.normal(key_z, (n, k), dtype=jnp.float64)
    means = jax.random.normal(key_mu, (c, k), dtype=jnp.float64) * 2
    covs = jnp.stack([random_spd(kk, k, condition=8.0) for kk in jax.random.split(key_cov, c)])
    priors = jax.random.uniform(key_pi, (c,), dtype=jnp.float64, minval=0.1)
    priors = priors / priors.sum()

    scores = np.asarray(discriminant_scores(z, means, covs, priors))
    constant = 0.5 * k * np.log(2 * np.pi)
    for j in range(c):
        expected = multivariate_normal.logpdf(
            np.asarray(z), np.asarray(means[j]), np.asarray(covs[j])
        ) + np.log(float(priors[j]))
        assert np.max(np.abs(scores[:, j] - constant - expected)) < 1e-9


def test_scores_are_invariant_to_an_invertible_reparametrization():
    """Sending z -> Lz with mu -> L mu and Sigma -> L Sigma L^T shifts every
    score by the same 2 log |det L|, so the argmax of eq. (5) does not move."""
    key = jax.random.PRNGKey(1)
    k, c, n = 4, 5, 40
    key_z, key_mu, key_cov, key_l = jax.random.split(key, 4)
    z = jax.random.normal(key_z, (n, k), dtype=jnp.float64)
    means = jax.random.normal(key_mu, (c, k), dtype=jnp.float64) * 2
    covs = jnp.stack([random_spd(kk, k, condition=6.0) for kk in jax.random.split(key_cov, c)])
    priors = jnp.full((c,), 1.0 / c, dtype=jnp.float64)
    lmap = jax.random.normal(key_l, (k, k), dtype=jnp.float64) + 2 * jnp.eye(k)

    base = discriminant_scores(z, means, covs, priors)
    mapped = discriminant_scores(
        z @ lmap.T, means @ lmap.T, lmap[None] @ covs @ lmap.T[None], priors
    )
    offset = np.asarray(mapped - base)
    assert np.ptp(offset) < 1e-8
    assert np.array_equal(np.asarray(predict(base)), np.asarray(predict(mapped)))


def test_confidence_is_the_max_softmax_probability():
    key = jax.random.PRNGKey(2)
    scores = jax.random.normal(key, (12, 6), dtype=jnp.float64) * 3
    exp = np.exp(np.asarray(scores) - np.asarray(scores).max(axis=1, keepdims=True))
    expected = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)
    assert np.max(np.abs(np.asarray(confidence(scores)) - expected)) < 1e-12
