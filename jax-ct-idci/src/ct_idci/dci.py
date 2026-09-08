"""Density-based DCI, iDCI, and Gaussian-copula CT+iDCI.

Equations follow Butler, Jiang, Silva, Hakula, Wildey,
"Copula Transformations for Data-Consistent Inversion" (arXiv:2609.02832).

The density-based DCI update is eq (5):

    pi_up(lambda) = pi_init(lambda) * pi_obs(phi(lambda)) / pi_pred(phi(lambda))

For iid samples from pi_init the r-values are that ratio. Under the
predictability assumption their sample mean is near 1 (eq 24).

iDCI (Algorithm 1) applies this update on each 1-d QoI subspace in turn,
re-estimating the predicted marginal from the current weights each time.

Theorem 3.2 (eq 14) factors the joint ratio into marginal ratios times a
copula ratio. After iDCI has matched the marginals, Theorem 3.4 / eq (18)
says the leftover update is only

    pi_CT(lambda) = pi_infty(lambda) * c_obs(U_obs(phi)) / c_infty(U_obs(phi))

evaluated in the observed PIT coordinates. Algorithm 2 multiplies the iDCI
r-values by that copula-density ratio. Theorem 3.5: an exact copula
transform recovers the original joint DCI solution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy import stats as jstats
from jax.scipy.linalg import solve_triangular

Array = jax.Array


def weighted_mean_cov(x: Array, weights: Array) -> tuple[Array, Array]:
    """Weighted mean and covariance of rows of x.

    Covariance is the probability-weighted outer product (no Bessel
    correction), matching the parametric Gaussian fit used in sec 6.1.1.
    """
    w = weights / jnp.sum(weights)
    mean = jnp.sum(w[:, None] * x, axis=0)
    xc = x - mean
    cov = (w[:, None] * xc).T @ xc
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def weighted_corr(x: Array, weights: Array) -> Array:
    """Correlation matrix of rows of x under probability weights."""
    _, cov = weighted_mean_cov(x, weights)
    std = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-18, None))
    corr = cov / (std[:, None] * std[None, :])
    corr = 0.5 * (corr + corr.T)
    # Keep a valid correlation: unit diagonal, clipped off-diagonals.
    d = corr.shape[0]
    corr = corr.at[jnp.diag_indices(d)].set(1.0)
    corr = jnp.clip(corr, -0.999, 0.999)
    corr = corr.at[jnp.diag_indices(d)].set(1.0)
    return corr


def gaussian_kl(mean_p: Array, cov_p: Array, mean_q: Array, cov_q: Array) -> Array:
    """KL(N(mean_p, cov_p) || N(mean_q, cov_q)), closed form."""
    d = mean_p.shape[-1]
    chol_q = jnp.linalg.cholesky(cov_q)
    # log det and quadratic via Cholesky, not a raw matmul of the
    # precision against a Gram matrix (TF32 on GPU would nibble that).
    logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_q)))
    sign_p, logdet_p = jnp.linalg.slogdet(cov_p)
    logdet_p = jnp.where(sign_p > 0, logdet_p, jnp.inf)
    # tr(cov_q^{-1} cov_p) = ||chol_q^{-1} chol_p||_F^2
    chol_p = jnp.linalg.cholesky(cov_p)
    m = solve_triangular(chol_q, chol_p, lower=True)
    trace = jnp.sum(m * m)
    delta = mean_p - mean_q
    y = solve_triangular(chol_q, delta, lower=True)
    quad = jnp.sum(y * y)
    return 0.5 * (trace + quad - d + logdet_q - logdet_p)


def dci_ratio(qoi: Array, obs_mean: Array, obs_scale: Array, pred_mean: Array, pred_scale: Array) -> Array:
    """1-d DCI r-factor pi_obs(q)/pi_pred(q) for Gaussian marginals.

    obs_scale / pred_scale are standard deviations. Used inside Algorithm 1.
    """
    log_obs = jstats.norm.logpdf(qoi, loc=obs_mean, scale=obs_scale)
    log_pred = jstats.norm.logpdf(qoi, loc=pred_mean, scale=pred_scale)
    return jnp.exp(log_obs - log_pred)


def joint_dci_ratio(qoi: Array, obs_mean: Array, obs_cov: Array, pred_mean: Array, pred_cov: Array) -> Array:
    """Joint (vector QoI) DCI ratio, eq (5) evaluated on samples of phi(lambda)."""
    log_obs = jstats.multivariate_normal.logpdf(qoi, mean=obs_mean, cov=obs_cov)
    log_pred = jstats.multivariate_normal.logpdf(qoi, mean=pred_mean, cov=pred_cov)
    return jnp.exp(log_obs - log_pred)


def iterative_dci(
    qoi: Array,
    obs_mean: Array,
    obs_var: Array,
    r0: Array | None = None,
    max_epoch: int = 40,
    tol: float = 1e-5,
) -> Array:
    """Algorithm 1, parametric Gaussian marginals.

    qoi: (n, d) predicted QoI samples from the initial measure.
    obs_mean, obs_var: (d,) observed 1-d Gaussian parameters.
    r0: optional starting r-values, default ones.

    Each subspace step normalizes r to unit sample mean, fits a weighted
    Gaussian to the current predicted marginal, and multiplies by the 1-d
    DCI ratio. Epochs stop when every marginal mean and variance matches
    the observed ones to ``tol`` relatively, or ``max_epoch`` is hit.
    """
    n, d = qoi.shape
    r = jnp.ones((n,), dtype=qoi.dtype) if r0 is None else r0
    obs_scale = jnp.sqrt(jnp.clip(obs_var, 1e-18, None))

    for _ in range(max_epoch):
        for i in range(d):
            r = r / jnp.mean(r)
            w = r / jnp.sum(r)
            qi = qoi[:, i]
            mu_p = jnp.sum(w * qi)
            var_p = jnp.sum(w * (qi - mu_p) ** 2)
            scale_p = jnp.sqrt(jnp.clip(var_p, 1e-18, None))
            r = r * dci_ratio(qi, obs_mean[i], obs_scale[i], mu_p, scale_p)
            r = jnp.clip(r, 0.0, None)

        # Marginal match diagnostic used as the epoch exit in Algorithm 1.
        w = r / jnp.sum(r)
        matched = True
        for i in range(d):
            qi = qoi[:, i]
            mu_p = jnp.sum(w * qi)
            var_p = jnp.sum(w * (qi - mu_p) ** 2)
            rel_m = jnp.abs(mu_p - obs_mean[i]) / (1.0 + jnp.abs(obs_mean[i]))
            rel_v = jnp.abs(var_p - obs_var[i]) / (1.0 + obs_var[i])
            if (rel_m > tol) or (rel_v > tol):
                matched = False
                break
        if matched:
            break
    return r


def gaussian_copula_density(u: Array, corr: Array) -> Array:
    """Gaussian copula density c(u; R).

    u is in (0, 1)^d. z = Phi^{-1}(u), then

        c(u; R) = phi_R(z) / prod_i phi(z_i)

    where phi_R is N(0, R). For R = I this is identically 1.
    """
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    z = jstats.norm.ppf(u)
    d = corr.shape[-1]
    log_num = jstats.multivariate_normal.logpdf(z, mean=jnp.zeros((d,), dtype=z.dtype), cov=corr)
    log_den = jnp.sum(jstats.norm.logpdf(z), axis=-1)
    return jnp.exp(log_num - log_den)


def copula_transform(
    qoi: Array,
    r: Array,
    obs_mean: Array,
    obs_var: Array,
    obs_corr: Array,
    pred_corr: Array | None = None,
) -> Array:
    """Eq (18): multiply iDCI r-values by c_obs(U_obs) / c_pred(U_obs).

    U_obs is the observed-marginal PIT of the predicted QoI samples, the
    common copula coordinate system of Theorem 3.3 after iDCI. If
    ``pred_corr`` is None it is the weighted correlation of those QoI
    samples (the parametric Gaussian copula of sec 6.1).
    """
    scale = jnp.sqrt(jnp.clip(obs_var, 1e-18, None))
    u = jstats.norm.cdf(qoi, loc=obs_mean, scale=scale)
    if pred_corr is None:
        pred_corr = weighted_corr(qoi, r)
    c_obs = gaussian_copula_density(u, obs_corr)
    c_pred = gaussian_copula_density(u, pred_corr)
    ratio = c_obs / jnp.clip(c_pred, 1e-18, None)
    out = r * ratio
    return jnp.clip(out, 0.0, None)


def ct_idci(
    qoi: Array,
    obs_mean: Array,
    obs_var: Array,
    obs_corr: Array,
    r0: Array | None = None,
    max_epoch: int = 40,
    n_ct: int = 1,
) -> tuple[Array, Array]:
    """Algorithm 2 with a Gaussian copula, one or more CT outer iterations.

    Returns (r_idci, r_ct). For Gaussians a single CT step is the exact
    recovery of Theorem 3.5; n_ct > 1 is the approximate iteration of
    Corollary 4.1.
    """
    r = iterative_dci(qoi, obs_mean, obs_var, r0=r0, max_epoch=max_epoch)
    r_idci = r
    for _ in range(n_ct):
        r = copula_transform(qoi, r, obs_mean, obs_var, obs_corr)
        if n_ct > 1:
            r = iterative_dci(qoi, obs_mean, obs_var, r0=r, max_epoch=max_epoch)
    return r_idci, r
