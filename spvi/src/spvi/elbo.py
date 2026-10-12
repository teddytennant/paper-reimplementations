"""Path-space objective pieces: Gaussian KL, kappa, quadrature, J."""

import jax.numpy as jnp


def diagonal_gaussian_kl(mu_q, v_q, mu_prior, v_prior):
    """KL between diagonal Gaussians, summed over the last axis (eq. 17).

    KL = 0.5 * sum_a [log(v_prior / v_q) + (v_q + (mu_q - mu_prior)^2) / v_prior - 1]
    """
    terms = (
        jnp.log(v_prior / v_q)
        + (v_q + (mu_q - mu_prior) ** 2) / v_prior
        - 1.0
    )
    return 0.5 * jnp.sum(terms, axis=-1)


def kl_standard_normal(mu_q, v_q):
    """KL(q || N(0, I)) = 0.5 * sum(-log v + v + mu^2 - 1)."""
    return 0.5 * jnp.sum(-jnp.log(v_q) + v_q + mu_q**2 - 1.0, axis=-1)


def kappa(lambda_q, lambda_theta, mu_q, v_q, mu_prior, v_prior):
    """Rate-mark integrand (eq. 8).

    kappa = lambda_q * log(lambda_q / lambda_theta) - lambda_q + lambda_theta
            + lambda_q * KL(q_U || p_theta(.|z))
    lambda_q is the posterior rate on the interval containing t (no z dependence).
    """
    kl = diagonal_gaussian_kl(mu_q, v_q, mu_prior, v_prior)
    return (
        lambda_q * jnp.log(lambda_q / lambda_theta)
        - lambda_q
        + lambda_theta
        + lambda_q * kl
    )


def stratified_times(t_left, t_right, uniforms):
    """Stratified quadrature nodes and weights (eq. 18).

    xi_{k,s} = t_{k-1} + (Delta_k / M) * (s - 1 + V_{k,s})
    weight_{k,s} = Delta_k / M

    t_left, t_right: (..., K)
    uniforms: (..., K, M) with entries in (0, 1)
    Returns xi, weights with the same shape as uniforms.
    Weights on each batch element sum to the sum of interval lengths.
    """
    probes = uniforms.shape[-1]
    delta = t_right - t_left
    strata = jnp.arange(1, probes + 1, dtype=uniforms.dtype)
    xi = t_left[..., None] + (delta[..., None] / probes) * (strata + uniforms - 1.0)
    weights = jnp.broadcast_to(delta[..., None] / probes, xi.shape)
    return xi, weights


def gaussian_loglik(y, mean, var):
    """Diagonal-Gaussian log-density, summed over coordinates and time.

    y, mean: (..., T, obs) or (T, obs). var broadcasts.
    The time axis is the second-to-last axis.
    """
    log_p = -0.5 * (jnp.log(2.0 * jnp.pi * var) + (y - mean) ** 2 / var)
    return jnp.sum(log_p, axis=(-1, -2))


def objective_j(loglik, kl_qp, lambda_q, delta, w_kl, beta_sparse):
    """One-trajectory objective (eq. 10).

    J = sum_k log p(y_k | Z_{t_k}) - w_KL * KL(Q||P) - beta_sparse * sum_k lambda_k * Delta_k

    kl_qp already includes KL(q0 || N(0, I)) + K_dyn.
    lambda_q and delta are reduced on the last axis.
    """
    sparsity = jnp.sum(lambda_q * delta, axis=-1)
    return loglik - w_kl * kl_qp - beta_sparse * sparsity


def frozen_path_objective(
    y,
    z_obs,
    mu0,
    v0,
    lambda_q,
    delta,
    quad_lambda_q,
    quad_lambda_theta,
    quad_mu_q,
    quad_v_q,
    quad_mu_prior,
    quad_v_prior,
    quad_weights,
    obs_var,
    w_kl,
    beta_sparse,
):
    """Assemble J on a frozen path (no resampling).

    Observation means are the leading observation coordinates of z_obs.
    Quadrature tensors are already evaluated at the pre-event states the
    caller chose. KL(Q||P) = KL(q0 || N(0, I)) + sum weights * kappa.
    """
    mean = z_obs[..., : y.shape[-1]]
    loglik = gaussian_loglik(y, mean, obs_var)
    kl0 = kl_standard_normal(mu0, v0)
    integrand = kappa(
        quad_lambda_q,
        quad_lambda_theta,
        quad_mu_q,
        quad_v_q,
        quad_mu_prior,
        quad_v_prior,
    )
    k_dyn = jnp.sum(quad_weights * integrand, axis=(-1, -2))
    return objective_j(loglik, kl0 + k_dyn, lambda_q, delta, w_kl, beta_sparse)
