"""The calibration claim: posterior width tracks the sampling variability.

Section 4 of arXiv:2609.01783 argues that the SME-BETEL posterior variance
matches the asymptotic sampling variance of the score matching estimator
whether or not the model is correct, so credible sets are valid confidence
sets. A posterior that instead treats Jhat as a log-likelihood has no such
guarantee.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sme_betel import models
from sme_betel.posterior import log_posterior_fn, metropolis_hastings
from sme_betel.scorematch import moment_fn, moment_matrix

# Working model: mean-zero Gaussian with unknown precision, a one-parameter
# unnormalized model. Data: a two-component Gaussian scale mixture standardized
# to unit variance, so the working model is wrong but theta* = 1 exactly.
MIX_SD = (1.0, 3.0)
MIX_P = 0.1
_VAR = (1 - MIX_P) * MIX_SD[0] ** 2 + MIX_P * MIX_SD[1] ** 2
THETA_STAR = 1.0


def _mixture(key, shape):
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, shape)
    heavy = jax.random.uniform(k2, shape) < MIX_P
    sd = jnp.where(heavy, MIX_SD[1], MIX_SD[0])
    return z * sd / jnp.sqrt(_VAR)


def _sm_estimate(x):
    """theta_hat_n = 1 / mean(x^2), the closed form for this model."""
    return 1.0 / jnp.mean(x**2, axis=-1)


def test_credible_width_tracks_sampling_variability_of_the_estimator():
    n = 400
    reps = 4000

    # Repeated-sample truth: the sampling standard deviation of theta_hat_n.
    xs = _mixture(jax.random.key(0), (reps, n))
    hats = _sm_estimate(xs)
    sd_repeat = float(jnp.std(hats))
    # theta_hat_n = 1/mean(x^2) is convex in mean(x^2), so its expectation sits
    # a little above theta* at finite n. About 2 percent here.
    assert abs(float(jnp.mean(hats)) - THETA_STAR) < 0.04

    log_q = models.gaussian_precision_log_q(1)
    gmat = moment_matrix(moment_fn(log_q))
    log_post = log_posterior_fn(gmat, lambda t: -0.5 * jnp.sum(t**2) / 100.0)

    def betel_sd(key):
        x = _mixture(jax.random.fold_in(key, 0), (n,))[:, None]
        start = _sm_estimate(x[:, 0])[None]
        out = metropolis_hastings(
            lambda t: log_post(t, x),
            start,
            jax.random.fold_in(key, 1),
            n_iter=15_000,
            burn_in=4_000,
            init_scale=sd_repeat,
        )
        return jnp.std(out.samples[:, 0]), jnp.mean(out.samples[:, 0]), start[0]

    sds, means, starts = jax.jit(jax.vmap(betel_sd))(
        jax.random.split(jax.random.key(1), 8)
    )
    sd_betel = float(jnp.mean(sds))

    # A posterior that reads Jhat as a log-likelihood. For this model
    # Jhat(a) = -a + 0.5 a^2 mean(x^2), so exp(-n Jhat) is exactly Gaussian with
    # variance theta_hat / n and no Laplace approximation is involved.
    sd_naive = float(jnp.mean(jnp.sqrt(starts / n)))

    # The posterior is centered where Theorem 1 says it is.
    assert float(jnp.max(jnp.abs(means - starts))) < 0.35 * sd_repeat

    # Loose but meaningful: the credible width lands within 30 percent of the
    # sampling spread, while reading Jhat as a log-likelihood undershoots it by
    # a factor near three. Over a six-seed sweep the first ratio stayed inside
    # 0.82 to 1.03 and the second inside 0.35 to 0.39.
    assert 0.7 < sd_betel / sd_repeat < 1.3
    assert sd_naive / sd_repeat < 0.5
    assert sd_betel / sd_naive > 1.8
