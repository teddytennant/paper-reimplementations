"""Online Laplace update of the last-layer Gaussian under a power prior.

Implements paper Proposition 1 / Algorithm 1 with W stored as (C, D) rather
than (D, C). Encoder is assumed frozen; only (mean, precision) change.

For a stream of labeled nodes, the update is:

  T <- lambda * T + eta * sum_v p_c(1-p_c) h_j^2 + eta * beta / Sigma_train
  G <- sum_v (y_oh - p) h^T - beta * (m - m_train) / Sigma_train
  m <- m + clip(eta * (1/T) * G, -delta, delta)

where p = softmax(m @ h / tau). The Hessian of the softmax-categorical is
the paper's diagonal approximation diag(p_c (1-p_c) h h^T).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _unpack(post):
    mean = jnp.asarray(post["mean"])
    if "precision" in post:
        precision = jnp.asarray(post["precision"])
    else:
        log_std = jnp.asarray(post["log_std"])
        precision = jnp.exp(-2.0 * jnp.clip(log_std, -20.0, 20.0))
    return mean, precision


def posterior_from_vbll(vbll_params):
    """Convert a trained VBLL dict {mean, log_std} into an online posterior."""
    mean, precision = _unpack(vbll_params)
    return pack_posterior(mean, precision)


def pack_posterior(mean, precision):
    precision = jnp.maximum(precision, 1e-12)
    return {
        "mean": mean,
        "precision": precision,
        "log_std": -0.5 * jnp.log(precision),
    }


def online_laplace_update(
    post,
    h,
    y,
    post_train,
    lambda_,
    eta=1.0,
    beta=1.0,
    clip_delta=1.0,
    eps=1e-6,
    temperature=1.0,
):
    """Power-prior Laplace step on the last-layer Gaussian.

    Parameters
    ----------
    post : dict
        Current posterior with `mean` (C, D) and `precision` or `log_std`.
    h : array (D,) or (B, D)
        Frozen encoder embeddings of the newly labeled nodes.
    y : int or array (B,)
        Observed class labels.
    post_train : dict
        Training posterior (m_train, T_train) used as the KL anchor.
    lambda_ : float in [0, 1)
        Exponential forgetting on the previous precision.
    eta, beta, clip_delta, eps, temperature
        Step size, anchor strength, mean-update clip, numerical floor, softmax tau.

    Returns
    -------
    dict with keys mean, precision, log_std.
    """
    mean, precision = _unpack(post)
    mean_train, precision_train = _unpack(post_train)
    h = jnp.atleast_2d(jnp.asarray(h))
    y = jnp.atleast_1d(jnp.asarray(y)).astype(jnp.int32)
    n_classes = mean.shape[0]

    logits = (h @ mean.T) / temperature
    probs = jax.nn.softmax(logits, axis=-1)
    y_oh = jax.nn.one_hot(y, n_classes, dtype=h.dtype)

    # Data gradient of log-likelihood wrt mean (C, D).
    g_data = (y_oh - probs).T @ h
    # Diagonal softmax Hessian / Fisher: sum_v p_c (1-p_c) h_j^2.
    fisher = (probs * (1.0 - probs)).T @ (h * h)

    var_train = 1.0 / (precision_train + eps)
    g_anchor = -beta * (mean - mean_train) / (var_train + eps)
    g = g_data + g_anchor

    precision_new = (
        lambda_ * precision + eta * fisher + eta * beta / (var_train + eps)
    )
    precision_new = jnp.maximum(precision_new, eps)
    var_new = 1.0 / precision_new
    delta = jnp.clip(eta * var_new * g, -clip_delta, clip_delta)
    mean_new = mean + delta
    return pack_posterior(mean_new, precision_new)
