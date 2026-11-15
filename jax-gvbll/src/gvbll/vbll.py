"""Fully factorized variational Bayesian last layer (VBLL / GVBLL).

Last-layer weights W have shape (C, D): C classes, D embedding dimensions.
q_phi(W) = prod_{j,c} N(m_{j,c}, sigma_{j,c}^2), reparameterized as
W = m + sigma * epsilon, epsilon ~ N(0, 1). Prior p(W) = N(0, sigma0^2 I).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_vbll(n_classes, embed_dim, prior_std=1.0):
    """Zero-mean last layer with std = prior_std (so initial KL is 0)."""
    mean = jnp.zeros((n_classes, embed_dim))
    log_std = jnp.full((n_classes, embed_dim), jnp.log(prior_std))
    return {"mean": mean, "log_std": log_std}


def std_from_log_std(log_std):
    return jnp.exp(jnp.clip(log_std, -20.0, 20.0))


def sample_weights(mean, std, key, n_samples):
    """Reparameterized draws. Returns W with shape (S, C, D)."""
    eps = jax.random.normal(key, (n_samples,) + tuple(mean.shape))
    return mean + std * eps


def kl_divergence(mean, std, prior_std=1.0):
    """Exact KL(q || p) for fully factorized Gaussians, q=N(m,s^2), p=N(0,s0^2)."""
    var = std**2
    prior_var = prior_std**2
    return 0.5 * jnp.sum(
        var / prior_var + (mean**2) / prior_var - 1.0 - jnp.log(var / prior_var)
    )


def logits_map(mean, h, temperature=1.0):
    """MAP logits: (N, C) from mean (C, D) and h (N, D)."""
    h = jnp.atleast_2d(h)
    return (h @ mean.T) / temperature


def predict_map(mean, h, temperature=1.0):
    """MAP class probabilities softmax(m @ h)."""
    return jax.nn.softmax(logits_map(mean, h, temperature=temperature), axis=-1)


def predict_mc(mean, std, h, key, n_samples=8, temperature=1.0):
    """Monte Carlo predictive: average softmax(W_s @ h) over S weight samples."""
    h = jnp.atleast_2d(h)
    weights = sample_weights(mean, std, key, n_samples)
    logits = jnp.einsum("scd,nd->snc", weights, h) / temperature
    return jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)


def uncertainty(probs):
    """Predictive entropy and max-probability from a (N, C) probability matrix."""
    probs = jnp.clip(probs, 1e-12, 1.0)
    entropy = -jnp.sum(probs * jnp.log(probs), axis=-1)
    max_prob = jnp.max(probs, axis=-1)
    return {"entropy": entropy, "max_prob": max_prob}
