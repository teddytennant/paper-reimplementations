"""ELBO and NLL for the variational Bayesian last layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gvbll.vbll import (
    kl_divergence,
    logits_map,
    sample_weights,
    std_from_log_std,
)


def expected_log_likelihood(mean, std, h, y, key, n_samples=8, temperature=1.0):
    """Monte Carlo E_q[sum_i log p(y_i | W, h_i)] with S softmax-categorical samples."""
    h = jnp.atleast_2d(h)
    y = jnp.atleast_1d(y).astype(jnp.int32)
    weights = sample_weights(mean, std, key, n_samples)
    logits = jnp.einsum("scd,nd->snc", weights, h) / temperature
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_py = jnp.take_along_axis(log_probs, y[None, :, None], axis=-1)[..., 0]
    return jnp.sum(jnp.mean(log_py, axis=0))


def elbo(
    mean,
    std,
    h,
    y,
    key,
    n_samples=8,
    beta=1.0,
    prior_std=1.0,
    temperature=1.0,
):
    """L = E_q[sum_i log p(y_i | W, h_i)] - beta * KL(q || p)."""
    ell = expected_log_likelihood(
        mean, std, h, y, key, n_samples=n_samples, temperature=temperature
    )
    kl = kl_divergence(mean, std, prior_std=prior_std)
    return ell - beta * kl


def neg_elbo(
    mean,
    std,
    h,
    y,
    key,
    n_samples=8,
    beta=1.0,
    prior_std=1.0,
    temperature=1.0,
):
    """Mean-NLL plus (beta / N) KL, matching paper eq. (4) up to the exact KL constants."""
    n = jnp.atleast_1d(y).shape[0]
    ell = expected_log_likelihood(
        mean, std, h, y, key, n_samples=n_samples, temperature=temperature
    )
    kl = kl_divergence(mean, std, prior_std=prior_std)
    return -ell / n + (beta * kl) / n


def map_nll(mean, h, y, temperature=1.0):
    """Deterministic mean negative log-likelihood under softmax(m @ h)."""
    y = jnp.atleast_1d(y).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(logits_map(mean, h, temperature=temperature), axis=-1)
    log_py = jnp.take_along_axis(log_probs, y[:, None], axis=-1)[..., 0]
    return -jnp.mean(log_py)


def vbll_neg_elbo(vbll_params, h, y, key, n_samples=8, beta=1.0, prior_std=1.0, temperature=1.0):
    std = std_from_log_std(vbll_params["log_std"])
    return neg_elbo(
        vbll_params["mean"],
        std,
        h,
        y,
        key,
        n_samples=n_samples,
        beta=beta,
        prior_std=prior_std,
        temperature=temperature,
    )
