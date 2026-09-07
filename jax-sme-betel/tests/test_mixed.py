"""Checks on the mixed-domain criterion of eq (2.9) and the weight of eq (S1)."""

import jax
import jax.numpy as jnp
import numpy as np

from sme_betel import models
from sme_betel.asymptotics import sandwich_covariance
from sme_betel.mixed import (
    boundary_weight_1d,
    box_weight,
    mixed_estimator,
    mixed_integrand,
    mixed_moment_fn,
    mixed_objective,
)
from sme_betel.posterior import log_posterior_fn, metropolis_hastings
from sme_betel.scorematch import score_integrand

EPS = 0.2
THETA_TRUE = jnp.array([1.5, -1.0, 0.8, 0.5, float(np.log(0.25))])


def test_boundary_weight_vanishes_at_the_boundary():
    """Eq (S1): h(a) = h(b) = 0, h = 1 on [a+eps, b-eps], h' = 0 at the ends."""
    a, b = -1.0, 2.0
    h = lambda x: boundary_weight_1d(jnp.asarray(x), a, b, EPS)
    dh = jax.grad(lambda x: boundary_weight_1d(x, a, b, EPS))

    assert float(h(a)) == 0.0
    assert float(h(b)) == 0.0
    assert abs(float(dh(jnp.asarray(a)))) < 1e-14
    assert abs(float(dh(jnp.asarray(b)))) < 1e-14

    interior = jnp.linspace(a + EPS, b - EPS, 41)
    assert float(jnp.max(jnp.abs(jax.vmap(h)(interior) - 1.0))) < 1e-14

    taper = jnp.linspace(a, a + EPS, 200)
    vals = jax.vmap(h)(taper)
    assert float(jnp.min(jnp.diff(vals))) >= 0.0
    assert float(jnp.min(vals)) >= 0.0 and float(jnp.max(vals)) <= 1.0

    # h is continuously differentiable where the taper meets the flat part.
    left = float(dh(jnp.asarray(a + EPS - 1e-9)))
    right = float(dh(jnp.asarray(a + EPS + 1e-9)))
    assert abs(left - right) < 1e-6


def test_box_weight_vanishes_on_every_face():
    """The rectangular form h(s) = prod_j h_j(s_j) from Appendix S3.1."""
    h = box_weight(jnp.zeros(2), jnp.ones(2), EPS)
    grid = jnp.linspace(0.0, 1.0, 21)
    for t in grid:
        for face in (
            jnp.array([0.0, float(t)]),
            jnp.array([1.0, float(t)]),
            jnp.array([float(t), 0.0]),
            jnp.array([float(t), 1.0]),
        ):
            assert float(h(face)) == 0.0
    assert abs(float(h(jnp.array([0.5, 0.5]))) - 1.0) < 1e-14


def test_mixed_integrand_matches_equation_2_9_written_out():
    """Compare against the four terms of eq (2.9) built separately.

    The library folds the second and third terms of eq (2.9) into one Laplacian
    over (y, s), which is valid because log p(s; theta) has no y-dependence.
    This rebuilds the terms the way the paper writes them.
    """
    h = box_weight(jnp.zeros(2), jnp.ones(2), EPS)
    theta = THETA_TRUE

    def log_cond(y, s, th):
        xi = th[0] * s[0] + th[1] * s[1]
        mu = th[3] + th[2] * xi
        return -0.5 * (y[0] - mu) ** 2 * jnp.exp(-th[4])

    def joint(z, th):
        return models.prefsamp_log_q(z[:1], z[1:], th)

    key = jax.random.key(0)
    ys = jax.random.normal(key, (7, 1))
    ss = jax.random.uniform(jax.random.key(1), (7, 2), minval=0.05, maxval=0.95)

    for y, s in zip(ys, ss):
        z = jnp.concatenate([y, s])
        psi = jax.grad(joint)(z, theta)
        hess_y = jax.hessian(lambda yy: log_cond(yy, s, theta))(y)
        hess_joint = jax.hessian(joint)(z, theta)
        lap_s = jnp.trace(hess_joint[1:, 1:])
        hs = h(s)
        grad_h = jax.grad(h)(s)
        literal = (
            0.5 * hs * jnp.dot(psi, psi)
            + hs * jnp.trace(hess_y)
            + hs * lap_s
            + jnp.dot(grad_h, psi[1:])
        )
        ours = mixed_integrand(models.prefsamp_log_q, h, y, s, theta)
        assert abs(float(ours - literal)) < 1e-11 * max(1.0, abs(float(literal)))


def test_unit_weight_gives_back_ordinary_score_matching():
    """With h identically one the boundary terms drop and (2.9) becomes (2.3)."""
    one = lambda s: jnp.array(1.0)
    theta = THETA_TRUE

    def joint(z, th):
        return models.prefsamp_log_q(z[:1], z[1:], th)

    ys = jax.random.normal(jax.random.key(2), (10, 1))
    ss = jax.random.uniform(jax.random.key(3), (10, 2))
    for y, s in zip(ys, ss):
        a = mixed_integrand(models.prefsamp_log_q, one, y, s, theta)
        b = score_integrand(joint, jnp.concatenate([y, s]), theta)
        assert abs(float(a - b)) < 1e-12 * max(1.0, abs(float(b)))


def test_with_no_bounded_component_it_is_plain_score_matching():
    """Drop s entirely and eq (2.9) collapses onto eq (2.3) on R^m."""
    empty = jnp.zeros((0,))
    one = lambda s: jnp.array(1.0)
    theta = jnp.array([1.3, 0.07, -0.4])

    def log_q_y(y, s, th):
        return models.quartic_log_q(y, th)

    xs = jax.random.normal(jax.random.key(4), (10, 3)) * 0.7
    for x in xs:
        a = mixed_integrand(log_q_y, one, x, empty, theta)
        b = score_integrand(models.quartic_log_q, x, theta)
        assert abs(float(a - b)) < 1e-12 * max(1.0, abs(float(b)))


def test_true_parameter_solves_the_mixed_population_moment_condition():
    """E[g_MD(y, s, theta_true)] = 0 when the working model is correct.

    Checked against the Monte Carlo standard error of the sample mean rather
    than against a fixed number, so the tolerance means something.
    """
    h = box_weight(jnp.zeros(2), jnp.ones(2), EPS)
    gmd = jax.jit(
        jax.vmap(mixed_moment_fn(models.prefsamp_log_q, h), in_axes=(0, 0, None))
    )
    n = 200_000
    y, s = models.sample_prefsamp(jax.random.key(5), n, THETA_TRUE, misspecified=False)
    gi = gmd(y, s, THETA_TRUE)
    mean = jnp.mean(gi, axis=0)
    stderr = jnp.std(gi, axis=0) / np.sqrt(n)
    assert float(jnp.max(jnp.abs(mean / stderr))) < 4.0


def test_mixed_moment_vanishes_at_the_mixed_estimator():
    """The eq (2.9) estimating equations hold exactly at the minimizer of J_MD,
    and the minimizer sits on the truth when the model is correctly specified."""
    h = box_weight(jnp.zeros(2), jnp.ones(2), EPS)
    jmd = mixed_objective(models.prefsamp_log_q, h)
    gmd = jax.jit(
        jax.vmap(mixed_moment_fn(models.prefsamp_log_q, h), in_axes=(0, 0, None))
    )
    n = 50_000
    y, s = models.sample_prefsamp(jax.random.key(6), n, THETA_TRUE, misspecified=False)
    theta = mixed_estimator(jmd, y, s, jnp.zeros(5))
    gi = gmd(y, s, theta)
    gbar = jnp.mean(gi, axis=0)
    assert float(jnp.max(jnp.abs(gbar))) < 1e-9 * float(jnp.max(jnp.abs(gi)))
    assert float(jnp.max(jnp.abs(theta - THETA_TRUE))) < 0.06


def test_mixed_posterior_covariance_matches_its_sandwich():
    """The eq (2.9) moments run all the way through eq (2.7) and Algorithm S1.

    Responses are Student t while the working model says Gaussian, so this is
    the misspecified mixed-domain case, and Theorem 1 still asks the posterior
    covariance to be n^{-1}(Gamma' Delta^{-1} Gamma)^{-1}.
    """
    n = 600
    h = box_weight(jnp.zeros(2), jnp.ones(2), EPS)
    jmd = mixed_objective(models.prefsamp_log_q, h)
    gmd = mixed_moment_fn(models.prefsamp_log_q, h)
    gmat = jax.vmap(gmd, in_axes=(0, 0, None))

    y, s = models.sample_prefsamp(jax.random.key(7), n, THETA_TRUE)
    theta_hat = mixed_estimator(jmd, y, s, jnp.zeros(5))
    stacked = jnp.concatenate([y, s], axis=1)
    target = sandwich_covariance(lambda z, th: gmd(z[:1], z[1:], th), stacked, theta_hat) / n

    log_post = log_posterior_fn(
        lambda data, th: gmat(data[0], data[1], th),
        lambda t: -0.5 * jnp.sum(t**2) / 25.0,
    )
    out = jax.jit(
        lambda k: metropolis_hastings(
            lambda t: log_post(t, (y, s)),
            theta_hat,
            k,
            n_iter=10_000,
            burn_in=3_000,
            step_chol=jnp.linalg.cholesky(target),
            init_scale=1.0,
        )
    )(jax.random.key(8))
    draws = np.asarray(out.samples)

    assert 0.2 < float(out.accept_rate) < 0.6
    tgt_sd = np.sqrt(np.diag(np.asarray(target)))
    post_sd = draws.std(axis=0)
    assert np.max(np.abs(post_sd / tgt_sd - 1.0)) < 0.3
    assert np.max(np.abs(draws.mean(0) - np.asarray(theta_hat)) / tgt_sd) < 0.4
