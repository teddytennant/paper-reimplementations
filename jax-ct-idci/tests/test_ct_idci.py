"""Properties from arXiv:2609.02832, mainly the linear-Gaussian example in 6.1.1."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats as jstats

from ct_idci.dci import (
    copula_transform,
    ct_idci,
    dci_ratio,
    gaussian_copula_density,
    gaussian_kl,
    iterative_dci,
    joint_dci_ratio,
    weighted_corr,
    weighted_mean_cov,
)

jax.config.update("jax_enable_x64", True)


MU = jnp.array([0.5, 0.5])
SIGMA = jnp.array([[0.5, 0.4], [0.4, 0.5]])
A_SKEW = jnp.array([[2.0, 1.0], [2.5, 0.5]])  # Q1, Q2
A_NON = jnp.array([[2.0, 1.0], [1.0, -1.0]])  # Q1, Q3


def _push_gaussian(mean, cov, A):
    return A @ mean, A @ cov @ A.T


def _sample_problem(key, A, n_init=10_000):
    k1, k2 = jax.random.split(key)
    lam = jax.random.normal(k1, (n_init, 2))
    qoi = lam @ A.T
    obs_mean, obs_cov = _push_gaussian(MU, SIGMA, A)
    obs_var = jnp.diag(obs_cov)
    std = jnp.sqrt(jnp.diag(obs_cov))
    obs_corr = obs_cov / (std[:, None] * std[None, :])
    pred_mean, pred_cov = _push_gaussian(jnp.zeros(2), jnp.eye(2), A)
    return {
        "lam": lam,
        "qoi": qoi,
        "obs_mean": obs_mean,
        "obs_cov": obs_cov,
        "obs_var": obs_var,
        "obs_corr": obs_corr,
        "pred_mean": pred_mean,
        "pred_cov": pred_cov,
        "A": A,
    }


def test_r_nonnegative_and_mean_near_one():
    """Eq (24): well-specified DCI r-values have sample mean near 1."""
    p = _sample_problem(jax.random.key(0), A_NON)
    r = joint_dci_ratio(p["qoi"], p["obs_mean"], p["obs_cov"], p["pred_mean"], p["pred_cov"])
    assert jnp.all(r >= 0)
    assert r.shape == (p["qoi"].shape[0],)
    np.testing.assert_allclose(float(jnp.mean(r)), 1.0, atol=0.05)


def test_oned_dci_matches_closed_form_gaussian():
    """A 1-d DCI update on a linear map recovers the closed-form updated Gaussian."""
    key = jax.random.key(1)
    lam = jax.random.normal(key, (20_000, 2))
    a = jnp.array([2.0, 1.0])
    q = lam @ a
    obs_mean = float(a @ MU)
    obs_var = float(a @ SIGMA @ a)
    pred_mean = 0.0
    pred_var = float(a @ a)
    r = dci_ratio(q, obs_mean, jnp.sqrt(obs_var), pred_mean, jnp.sqrt(pred_var))
    wmean, wcov = weighted_mean_cov(lam, r)
    # Push-forward of the updated samples must match the observed 1-d Gaussian.
    q_mean = float(wmean @ a)
    q_var = float(a @ wcov @ a)
    np.testing.assert_allclose(q_mean, obs_mean, atol=0.03)
    np.testing.assert_allclose(q_var, obs_var, rtol=0.05)


def test_gaussian_copula_identity_is_one():
    """R = I: copula density is 1, so CT is a no-op."""
    u = jnp.array([[0.1, 0.7], [0.4, 0.4], [0.9, 0.2]])
    c = gaussian_copula_density(u, jnp.eye(2))
    np.testing.assert_allclose(np.array(c), np.ones(3), atol=1e-10)
    qoi = jstats.norm.ppf(u)
    r = jnp.array([0.5, 1.2, 0.8])
    r_ct = copula_transform(
        qoi,
        r,
        obs_mean=jnp.zeros(2),
        obs_var=jnp.ones(2),
        obs_corr=jnp.eye(2),
        pred_corr=jnp.eye(2),
    )
    np.testing.assert_allclose(np.array(r_ct), np.array(r), atol=1e-10)


def test_idci_matches_observed_marginals_skewed():
    """After iDCI, each 1-d push-forward matches the observed marginal."""
    p = _sample_problem(jax.random.key(2), A_SKEW)
    r = iterative_dci(p["qoi"], p["obs_mean"], p["obs_var"], max_epoch=50)
    w = r / jnp.sum(r)
    for i in range(2):
        qi = p["qoi"][:, i]
        mu = float(jnp.sum(w * qi))
        var = float(jnp.sum(w * (qi - mu) ** 2))
        np.testing.assert_allclose(mu, float(p["obs_mean"][i]), atol=0.03)
        np.testing.assert_allclose(var, float(p["obs_var"][i]), rtol=0.05)


def test_skewed_copula_drops_kl_by_an_order():
    """Sec 6.1.1: skewed (Q1, Q2), iDCI leaves large KL, CT recovers DG."""
    p = _sample_problem(jax.random.key(3), A_SKEW)
    r_idci, r_ct = ct_idci(
        p["qoi"], p["obs_mean"], p["obs_var"], p["obs_corr"], max_epoch=50, n_ct=1
    )
    m_i, c_i = weighted_mean_cov(p["lam"], r_idci)
    m_c, c_c = weighted_mean_cov(p["lam"], r_ct)
    kl_idci = float(gaussian_kl(m_i, c_i, MU, SIGMA))
    kl_ct = float(gaussian_kl(m_c, c_c, MU, SIGMA))
    assert kl_idci > 0.1, f"iDCI leftover KL too small: {kl_idci}"
    assert kl_ct < 0.1 * kl_idci, f"CT did not drop KL enough: {kl_idci} -> {kl_ct}"
    assert kl_ct < 0.05


def test_nonskewed_idci_already_close():
    """Sec 6.1.1: non-skewed (Q1, Q3), iDCI is already close; CT is not worse."""
    p = _sample_problem(jax.random.key(4), A_NON)
    r_idci, r_ct = ct_idci(
        p["qoi"], p["obs_mean"], p["obs_var"], p["obs_corr"], max_epoch=50, n_ct=1
    )
    m_i, c_i = weighted_mean_cov(p["lam"], r_idci)
    m_c, c_c = weighted_mean_cov(p["lam"], r_ct)
    kl_idci = float(gaussian_kl(m_i, c_i, MU, SIGMA))
    kl_ct = float(gaussian_kl(m_c, c_c, MU, SIGMA))
    assert kl_idci < 0.05
    assert kl_ct <= kl_idci + 0.01


def test_theorem_35_ct_recovers_joint_dci():
    """Theorem 3.5: exact copula transform of iDCI recovers joint DCI."""
    p = _sample_problem(jax.random.key(5), A_SKEW)
    r_idci, r_ct = ct_idci(
        p["qoi"], p["obs_mean"], p["obs_var"], p["obs_corr"], max_epoch=50, n_ct=1
    )
    r_joint = joint_dci_ratio(
        p["qoi"], p["obs_mean"], p["obs_cov"], p["pred_mean"], p["pred_cov"]
    )
    # Compare as probability weights (scale-invariant).
    w_ct = r_ct / jnp.sum(r_ct)
    w_j = r_joint / jnp.sum(r_joint)
    tv = 0.5 * float(jnp.sum(jnp.abs(w_ct - w_j)))
    assert tv < 0.02, f"TV between CT and joint DCI weights: {tv}"
    m_ct, c_ct = weighted_mean_cov(p["lam"], r_ct)
    m_j, c_j = weighted_mean_cov(p["lam"], r_joint)
    kl = float(gaussian_kl(m_ct, c_ct, m_j, c_j))
    assert kl < 1e-3, f"KL(CT || joint DCI) = {kl}"


def test_weighted_corr_identity_on_independent():
    x = jax.random.normal(jax.random.key(6), (8000, 2))
    r = jnp.ones((8000,))
    corr = weighted_corr(x, r)
    np.testing.assert_allclose(np.array(corr), np.eye(2), atol=0.05)
