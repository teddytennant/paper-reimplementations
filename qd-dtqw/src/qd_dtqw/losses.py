"""Training losses and histogram KL (eqs. 12–13)."""

from __future__ import annotations

import jax.numpy as jnp


def log_softmax(logits, axis: int = -1):
    logits = jnp.asarray(logits)
    shifted = logits - jnp.max(logits, axis=axis, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))


def softmax(logits, axis: int = -1):
    return jnp.exp(log_softmax(logits, axis=axis))


def categorical_cross_entropy(logits, targets, k: int | None = None):
    """Mean -sum_x y_x log p_x over batch/spatial axes (eq. 12 CE term).

    `targets` are integer class labels in {0,...,K-1}.
    """
    logits = jnp.asarray(logits)
    targets = jnp.asarray(targets)
    if k is None:
        k = logits.shape[-1]
    log_p = log_softmax(logits, axis=-1)
    n = targets.size
    one_hot = jnp.eye(k, dtype=log_p.dtype)[targets.reshape(-1)]
    return -jnp.sum(one_hot * log_p.reshape(n, k)) / jnp.maximum(n, 1)


def intensity_histogram(images, k: int, axis=None):
    """Normalized occupancy over the K intensity bins (eq. 13)."""
    images = jnp.asarray(images).astype(jnp.int32).reshape(-1)
    counts = jnp.bincount(images, length=k).astype(jnp.float64)
    total = jnp.maximum(counts.sum(), 1.0)
    return counts / total


def histogram_kl(p, q, eps: float = 1e-12):
    """KL(P || Q) = sum_x P(x) log(P(x)/Q(x)) (eq. 13)."""
    p = jnp.asarray(p, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    p = p / jnp.maximum(p.sum(), eps)
    q = q / jnp.maximum(q.sum(), eps)
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.sum(p * jnp.log(p / q))


def time_alpha(t, T: int):
    """α_t large at small t (reconstruct), small at large t (match histogram).

    The paper scales α_t with t so high t matches the intensity histogram
    and low t reconstructs x_t. That is α_t = 1 - t / (T-1).
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    denom = jnp.maximum(float(T) - 1.0, 1.0)
    return 1.0 - jnp.clip(t / denom, 0.0, 1.0)


def mixed_loss(logits, targets, t, T: int, k: int | None = None):
    """Eq. 12: α_t CE + (1-α_t) KL(true hist || predicted hist)."""
    logits = jnp.asarray(logits)
    if k is None:
        k = logits.shape[-1]
    ce = categorical_cross_entropy(logits, targets, k=k)
    true_hist = intensity_histogram(targets, k)
    pred_hist = softmax(logits, axis=-1).reshape(-1, k).mean(axis=0)
    kl = histogram_kl(true_hist, pred_hist)
    a = time_alpha(t, T)
    return a * ce + (1.0 - a) * kl
