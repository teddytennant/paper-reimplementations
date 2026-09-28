"""Exact Gaussian / mixture scores, Tweedie, and the affine denoiser (eq. 3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from intrinsic_atlas.schedules import VPSchedule

# Figure 4 / Table 2 2D Gaussian population (eigen-variances of C).
FIG4_LAMBDAS = (0.084, 0.716)
FIG4_T = 1000
FIG4_RHO_PRODUCTS = (3.4e-6, 2.9e-5)


def _as_cov(cov) -> jnp.ndarray:
    cov = jnp.asarray(cov)
    if cov.ndim == 1:
        return jnp.diag(cov)
    return cov


def _solve(cov: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """Solve cov x = vec for vec of shape (D,) or (..., D)."""
    vec = jnp.asarray(vec)
    if vec.ndim == 1:
        return jnp.linalg.solve(cov, vec)
    flat = vec.reshape(-1, vec.shape[-1])
    solved = jax.vmap(lambda v: jnp.linalg.solve(cov, v))(flat)
    return solved.reshape(vec.shape)


def noised_mean_cov(mu, cov, alpha_bar):
    """Marginal of x_t: N(√ᾱ μ, ᾱ C + (1-ᾱ) I)."""
    mu = jnp.asarray(mu)
    cov = _as_cov(cov)
    dim = cov.shape[0]
    mean = jnp.sqrt(alpha_bar) * mu
    sigma = alpha_bar * cov + (1.0 - alpha_bar) * jnp.eye(dim, dtype=cov.dtype)
    return mean, sigma


def gaussian_score(x_t, alpha_bar, mu, cov):
    """∇_{x_t} log f_{X_t}(x_t) for x_0 ~ N(μ, C)."""
    mean, sigma = noised_mean_cov(mu, cov, alpha_bar)
    return -_solve(sigma, x_t - mean)


def tweedie_x0(x_t, alpha_bar, score):
    """Standard VP-SDE / DDPM Tweedie posterior mean E[x_0 | x_t].

    x̂0 = (x_t + (1 - ᾱ_t) ∇ log f_{X_t}(x_t)) / √ᾱ_t

    This is the formula that recovers E[x_0|x_t] for a Gaussian and matches
    the affine denoiser (eq. 3). The PDF layout of eq. (2) is ambiguous.
    """
    return (x_t + (1.0 - alpha_bar) * score) / jnp.sqrt(alpha_bar)


def tweedie_x0_from_eps(x_t, alpha_bar, eps):
    """x̂0 from a noise prediction: x̂0 = (x_t - √(1-ᾱ_t) ε̂) / √ᾱ_t."""
    return (x_t - jnp.sqrt(1.0 - alpha_bar) * eps) / jnp.sqrt(alpha_bar)


def affine_denoiser(x_t, alpha_bar, mu, cov):
    """Closed-form affine denoiser (eq. 3).

    x̂0(x_t) = μ + √ᾱ_t C (ᾱ_t C + (1-ᾱ_t) I)^{-1} (x_t - √ᾱ_t μ)
    """
    mu = jnp.asarray(mu)
    cov = _as_cov(cov)
    mean, sigma = noised_mean_cov(mu, cov, alpha_bar)
    delta = x_t - mean
    return mu + jnp.sqrt(alpha_bar) * (cov @ _solve(sigma, delta).T).T


def clean_deviation(x_t, alpha_bar, mu):
    """Deviation in clean-image units: e_t = (x_t - √ᾱ_t μ) / √ᾱ_t."""
    return (x_t - jnp.sqrt(alpha_bar) * mu) / jnp.sqrt(alpha_bar)


def mixture_score(x_t, alpha_bar, mus, cov, log_weights=None):
    """Score of an equal-covariance Gaussian mixture (unconditioned).

    mus: (K, D). Shared covariance ``cov``. Responsibilities are the
    posterior component weights of the noised marginal.
    """
    mus = jnp.asarray(mus)
    cov = _as_cov(cov)
    k = mus.shape[0]
    if log_weights is None:
        log_weights = jnp.zeros((k,), dtype=mus.dtype)
    else:
        log_weights = jnp.asarray(log_weights)

    mean_k, sigma = noised_mean_cov(mus[0], cov, alpha_bar)
    # Recompute per-component means; Sigma_t is shared.
    means = jnp.sqrt(alpha_bar) * mus  # (K, D)

    def log_n(x, mean):
        diff = x - mean
        quad = jnp.sum(diff * _solve(sigma, diff), axis=-1)
        sign, logdet = jnp.linalg.slogdet(sigma)
        dim = mean.shape[-1]
        return -0.5 * (quad + logdet + dim * jnp.log(2.0 * jnp.pi))

    # log_n over components: (K, batch) or (K,)
    log_ps = jax.vmap(lambda m: log_n(x_t, m))(means)  # (K,) or (K, B)
    logits = log_ps + log_weights.reshape((-1,) + (1,) * (log_ps.ndim - 1))
    log_z = jax.nn.logsumexp(logits, axis=0)
    pi = jnp.exp(logits - log_z)  # (K,) or (K, B)

    scores_k = jax.vmap(lambda m: -_solve(sigma, x_t - m))(means)  # (K, ..., D)
    # Weighted sum over K.
    pi_b = jnp.expand_dims(pi, -1)
    return jnp.sum(pi_b * scores_k, axis=0)


def make_gaussian_x0_fn(mu, cov, schedule: VPSchedule):
    """predict_x0(x_t, t) via Tweedie on the exact Gaussian score."""
    mu = jnp.asarray(mu)
    cov = _as_cov(cov)

    def predict_x0(x_t, t):
        alpha_bar = schedule.alphas_cumprod[t]
        score = gaussian_score(x_t, alpha_bar, mu, cov)
        return tweedie_x0(x_t, alpha_bar, score)

    return predict_x0


def make_mixture_x0_fn(mus, cov, schedule: VPSchedule, log_weights=None):
    """Unconditioned mixture Tweedie denoiser."""
    mus = jnp.asarray(mus)
    cov = _as_cov(cov)

    def predict_x0(x_t, t):
        alpha_bar = schedule.alphas_cumprod[t]
        score = mixture_score(x_t, alpha_bar, mus, cov, log_weights=log_weights)
        return tweedie_x0(x_t, alpha_bar, score)

    return predict_x0


def make_conditioned_x0_fn(mus, cov, schedule: VPSchedule, component: int):
    """Condition on mixture component ``component`` (Table 2, last row)."""
    mu = jnp.asarray(mus)[component]
    return make_gaussian_x0_fn(mu, cov, schedule)
