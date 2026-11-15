"""The DPD objective, its alpha -> 0 limit, and its stationarity conditions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mkmeans_dpd import (
    dpd_gradients,
    dpd_objective,
    joint_objective,
    mahalanobis,
    mdpde,
    pointwise_loss,
    spec_from_cov,
)
from mkmeans_dpd.dpd import _a_const


def _one_cluster(x, sigma=None):
    x = jnp.asarray(x)
    mu = jnp.mean(x, axis=0)[None, :]
    if sigma is None:
        sigma = jnp.cov(x.T, bias=True)[None]
    spec = spec_from_cov(jnp.asarray(sigma), 1e-9)
    resp = jnp.ones((x.shape[0], 1))
    return mu, spec, resp


def test_objective_is_the_average_of_the_pointwise_loss(small_cluster):
    """Equation (3.7): V_alpha is exactly the mean of v_alpha over the cluster."""
    x = jnp.asarray(small_cluster)
    mu, spec, resp = _one_cluster(x)
    b = mahalanobis(x, mu, spec)
    v = pointwise_loss(b, spec.logdet, 0.3, x.shape[1])
    big_v = dpd_objective(b, spec.logdet, resp, 0.3, x.shape[1])
    np.testing.assert_allclose(np.asarray(big_v)[0], float(jnp.mean(v)), rtol=1e-12)
    # and Q of Equation (3.13) is n_k V_k
    q = joint_objective(b, spec.logdet, resp, 0.3, x.shape[1])
    np.testing.assert_allclose(float(q), x.shape[0] * float(big_v[0]), rtol=1e-12)


def test_objective_is_jittable(small_cluster):
    x = jnp.asarray(small_cluster)
    mu, spec, resp = _one_cluster(x)
    b = mahalanobis(x, mu, spec)
    f = jax.jit(joint_objective, static_argnums=(3, 4))
    np.testing.assert_allclose(
        float(f(b, spec.logdet, resp, 0.3, x.shape[1])),
        float(joint_objective(b, spec.logdet, resp, 0.3, x.shape[1])),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------
# alpha -> 0
# --------------------------------------------------------------------------


def _gaussian_nll(x, mu, spec):
    """-log N(x; mu, Sigma) = (p log 2pi + log|Sigma| + B) / 2."""
    p = x.shape[1]
    b = mahalanobis(x, mu, spec)
    return 0.5 * (p * jnp.log(2 * jnp.pi) + spec.logdet[None, :] + b)


@pytest.mark.parametrize("alpha", [1e-3, 1e-4, 1e-5])
def test_pointwise_loss_tends_to_the_gaussian_negative_log_likelihood(
    small_cluster, alpha
):
    """v_alpha + 1/alpha -> -log f(x; mu, Sigma) as alpha -> 0, with an O(alpha) error.

    Expanding Equation (3.8) about alpha = 0 gives
    v_alpha = -1/alpha + (p log 2pi + log|Sigma| + B)/2 + O(alpha). The -1/alpha
    is free of k, so the assignment rule of Algorithm 3 reduces in the limit to
    the Gaussian maximum-likelihood assignment, which is Lapidot's repair of
    Mahalanobis K-means that Section 3.3 says it generalizes.
    """
    x = jnp.asarray(small_cluster)
    sigma = jnp.stack([jnp.cov(x.T, bias=True), 2.0 * jnp.eye(2)])
    mu = jnp.stack([jnp.mean(x, axis=0), jnp.mean(x, axis=0) + 1.0])
    spec = spec_from_cov(sigma, 1e-9)

    v = pointwise_loss(mahalanobis(x, mu, spec), spec.logdet, alpha, 2)
    err = jnp.max(jnp.abs(v + 1.0 / alpha - _gaussian_nll(x, mu, spec)))
    # The expansion's next term is O(alpha), so the error must shrink linearly.
    assert float(err) < 60.0 * alpha


def test_assignment_reduces_to_gaussian_ml_at_small_alpha(small_cluster):
    """At small alpha, arg min_k v_alpha and arg min_k NLL pick the same cluster."""
    rng = np.random.default_rng(3)
    x = jnp.asarray(np.concatenate([small_cluster, rng.normal(size=(200, 2)) * 4]))
    sigma = jnp.stack(
        [jnp.array([[2.0, 0.9], [0.9, 1.5]]), jnp.array([[6.0, -2.0], [-2.0, 3.0]])]
    )
    mu = jnp.array([[1.0, -2.0], [0.5, 0.5]])
    spec = spec_from_cov(sigma, 1e-9)
    v = pointwise_loss(mahalanobis(x, mu, spec), spec.logdet, 1e-6, 2)
    assert bool(
        jnp.all(jnp.argmin(v, axis=1) == jnp.argmin(_gaussian_nll(x, mu, spec), axis=1))
    )


def test_assignment_reduces_to_raw_mahalanobis_when_the_covariances_are_equal(
    small_cluster,
):
    """With one shared Sigma the log|Sigma| term is common, so Algorithm 3's
    assignment collapses onto Algorithm 2's raw Mahalanobis rule (Step 4)."""
    x = jnp.asarray(small_cluster)
    sigma = jnp.broadcast_to(jnp.array([[2.0, 0.9], [0.9, 1.5]]), (3, 2, 2))
    mu = jnp.array([[0.0, 0.0], [1.0, -2.0], [3.0, 1.0]])
    spec = spec_from_cov(sigma, 1e-9)
    b = mahalanobis(x, mu, spec)
    v = pointwise_loss(b, spec.logdet, 0.4, 2)
    assert bool(jnp.all(jnp.argmin(v, axis=1) == jnp.argmin(b, axis=1)))


@pytest.mark.parametrize("alpha", [1e-4, 1e-6])
def test_weighted_update_reduces_to_the_mle(small_cluster, alpha):
    """Equation (3.6) at alpha -> 0 is the Step 5 update of Algorithm 1.

    The weights omega_i all go to 1 and the correction m alpha / (1+alpha)^{(p+2)/2}
    goes to 0, so the mean becomes the sample mean and the covariance the MLE.
    """
    from mkmeans_dpd import weighted_step

    x = jnp.asarray(small_cluster)
    mu, spec, resp = _one_cluster(x)
    mu_new, spec_new = weighted_step(x, mu, spec, resp, alpha, 1e-12)
    np.testing.assert_allclose(
        np.asarray(mu_new)[0], np.asarray(jnp.mean(x, axis=0)), atol=20 * alpha
    )
    np.testing.assert_allclose(
        np.asarray(spec_new.sigma)[0],
        np.asarray(jnp.cov(x.T, bias=True)),
        rtol=20 * alpha,
    )


def test_the_correction_term_matches_the_paper(small_cluster):
    """m alpha / (1+alpha)^{(p+2)/2} of Equation (3.9) equals m / A of (3.8)."""
    alpha, p, m = 0.35, 2, 400
    a = float(_a_const(alpha, p))
    np.testing.assert_allclose(
        m * alpha / (1 + alpha) ** ((p + 2) / 2), m / a, rtol=1e-12
    )


# --------------------------------------------------------------------------
# The update is a stationary point of the objective
# --------------------------------------------------------------------------


def _v_of(x, mu_flat, lam_flat, resp, alpha, p, k):
    """V_k as a plain function of (mu, Lambda), for finite differences."""
    mu = mu_flat.reshape(k, p)
    lam = lam_flat.reshape(k, p, p)
    sigma = jnp.linalg.inv(0.5 * (lam + jnp.swapaxes(lam, -1, -2)))
    spec = spec_from_cov(sigma, 0.0)
    return dpd_objective(mahalanobis(x, mu, spec), spec.logdet, resp, alpha, p)


def test_gradients_match_finite_differences(small_cluster):
    """Equations (3.10) and (3.11) against central differences of Equation (3.5).

    Checked as directional derivatives along a symmetric perturbation of Lambda,
    which is the convention the paper's matrix derivative is written in.
    """
    alpha = 0.4
    x = jnp.asarray(small_cluster)
    p, k = 2, 1
    mu, spec, resp = _one_cluster(x)
    # Deliberately off the fixed point, so the gradient is not near zero.
    mu = mu + 0.3
    spec = spec_from_cov(spec.sigma * 1.4, 0.0)

    g_mu, g_lam = dpd_gradients(x, mu, spec, resp, alpha)

    rng = np.random.default_rng(0)
    d_mu = jnp.asarray(rng.normal(size=(k, p)))
    e = rng.normal(size=(k, p, p))
    d_lam = jnp.asarray(0.5 * (e + np.swapaxes(e, -1, -2)))

    eps = 1e-6
    lam = spec.lam
    hi = _v_of(
        x, (mu + eps * d_mu).ravel(), (lam + eps * d_lam).ravel(), resp, alpha, p, k
    )
    lo = _v_of(
        x, (mu - eps * d_mu).ravel(), (lam - eps * d_lam).ravel(), resp, alpha, p, k
    )
    numeric = float((hi - lo)[0] / (2 * eps))
    analytic = float(jnp.sum(g_mu * d_mu) + jnp.sum(g_lam * d_lam))
    np.testing.assert_allclose(numeric, analytic, rtol=1e-6)


def test_the_weighted_fixed_point_has_zero_gradient(small_cluster):
    """The estimating equations (3.6) are exactly grad V = 0, Section 3.3.

    Run the weighted iteration to convergence on one cluster and check the
    analytic gradients vanish there, measured against the gradient at a nearby
    point so the bound does not depend on the units of the data.
    """
    alpha = 0.5
    x = jnp.asarray(small_cluster)
    mu0, spec0, resp = _one_cluster(x)
    mu, spec = mdpde(x, mu0, spec0, resp, alpha, 1e-12, n_inner=400)

    g_mu, g_lam = dpd_gradients(x, mu, spec, resp, alpha)
    at_fit = float(jnp.sqrt(jnp.sum(g_mu**2) + jnp.sum(g_lam**2)))

    off_mu, off_lam = dpd_gradients(
        x, mu + 0.25, spec_from_cov(spec.sigma * 1.25, 0.0), resp, alpha
    )
    nearby = float(jnp.sqrt(jnp.sum(off_mu**2) + jnp.sum(off_lam**2)))

    assert at_fit < 1e-8 * nearby


def test_the_fitted_model_is_stationary(small_cluster):
    """Same check at the end of a full DC-MK-means DPD fit, per cluster."""
    from mkmeans_dpd import dc_mkmeans_dpd
    from mkmeans_dpd.baselines import _one_hot
    from tests.conftest import elliptical

    x, _, means, covs = elliptical(11, n_per=250)
    alpha = 0.4
    res = dc_mkmeans_dpd(x, 3, alpha=alpha, centers=means, covariances=covs)
    resp = _one_hot(res.labels, 3)
    spec = spec_from_cov(res.covariances, 1e-12)
    g_mu, g_lam = dpd_gradients(jnp.asarray(x), res.centers, spec, resp, alpha)

    off = spec_from_cov(res.covariances * 1.25, 1e-12)
    o_mu, o_lam = dpd_gradients(jnp.asarray(x), res.centers + 0.25, off, resp, alpha)

    for k in range(3):
        at_fit = float(jnp.sqrt(jnp.sum(g_mu[k] ** 2) + jnp.sum(g_lam[k] ** 2)))
        nearby = float(jnp.sqrt(jnp.sum(o_mu[k] ** 2) + jnp.sum(o_lam[k] ** 2)))
        assert at_fit < 1e-5 * nearby


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------


def test_quadratic_form_is_not_computed_in_tf32():
    """Mahalanobis distances in float32 must stay within float32, not TF32.

    A float32 matmul on an H200 defaults to TF32 (10 explicit mantissa bits,
    around 5e-4 relative). Cluster assignment compares distances that can differ
    by less than that, so the argmin flips. ``mahalanobis`` pins the matmul to
    Precision.HIGHEST; this checks the float32 result against a float64
    reference at a tolerance TF32 could not meet.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    sigma = a @ a.T + 4 * np.eye(4)
    x = rng.normal(size=(512, 4)) * 3.0
    mu = rng.normal(size=(3, 4))

    ref = mahalanobis(
        jnp.asarray(x, jnp.float64),
        jnp.asarray(mu, jnp.float64),
        spec_from_cov(
            jnp.asarray(sigma[None] * np.ones((3, 1, 1)), jnp.float64), 1e-12
        ),
    )
    got = mahalanobis(
        jnp.asarray(x, jnp.float32),
        jnp.asarray(mu, jnp.float32),
        spec_from_cov(
            jnp.asarray(sigma[None] * np.ones((3, 1, 1)), jnp.float32), 1e-12
        ),
    )
    rel = float(jnp.max(jnp.abs(got.astype(jnp.float64) - ref) / jnp.abs(ref)))
    assert rel < 3e-6, f"relative error {rel:.2e} is in TF32 territory"
