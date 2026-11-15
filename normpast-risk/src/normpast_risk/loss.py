"""Classification, weakly supervised segment, and healthy-only normative losses.

Segment loss is a concrete weakly supervised reading, not a quoted equation.
An Alzheimer's sample (y = 1) contributes -log(max_j p_j + 1e-8) plus the
mean binary entropy of p_j. A healthy sample (y = 0) contributes the mean
over segments of -log(1 - p_j + 1e-8). The sample head is trained with
binary cross-entropy on the sigmoid of an attention pool of pre-sigmoid
segment logits.

L = L_cls + lambda_seg * L_seg + lambda_norm * L_norm
with lambda_seg = 0.1 and lambda_norm = 0.2.
"""

from __future__ import annotations

import jax.numpy as jnp


LAMBDA_SEG = 0.1
LAMBDA_NORM = 0.2
LOG_EPS = 1e-8


def classification_loss(sample_prob, y, eps: float = LOG_EPS):
    """Mean binary cross-entropy between p_hat and y."""
    p = jnp.clip(jnp.asarray(sample_prob), eps, 1.0 - eps)
    labels = jnp.asarray(y, dtype=p.dtype).reshape((-1,))
    per = -(labels * jnp.log(p) + (1.0 - labels) * jnp.log(1.0 - p))
    return jnp.mean(per)


def binary_entropy(p, eps: float = LOG_EPS):
    clipped = jnp.clip(p, eps, 1.0 - eps)
    return -(clipped * jnp.log(clipped) + (1.0 - clipped) * jnp.log(1.0 - clipped))


def segment_loss(segment_risk, y, eps: float = LOG_EPS):
    """Weakly supervised segment loss, mean over the batch.

    Alzheimer's samples are pushed to put probability on some segment and to
    be confident. Healthy samples are pushed so every segment looks healthy.
    """
    p = jnp.asarray(segment_risk)
    labels = jnp.asarray(y, dtype=p.dtype).reshape((-1,))
    max_p = jnp.max(p, axis=-1)
    ad = -jnp.log(max_p + eps) + jnp.mean(binary_entropy(p), axis=-1)
    healthy = jnp.mean(-jnp.log(1.0 - p + eps), axis=-1)
    per = jnp.where(labels > 0.5, ad, healthy)
    return jnp.mean(per)


def normative_loss(hat_next, target_next, y):
    """Mean squared error of next-token prediction, healthy samples only.

    AD samples are masked out of the mean. If the batch has no healthy
    sample, or there is no next token, the loss is 0.
    hat_next predicts target_next, both shaped (batch, S-1, d).
    """
    hat_next = jnp.asarray(hat_next)
    target_next = jnp.asarray(target_next)
    labels = jnp.asarray(y, dtype=hat_next.dtype).reshape((-1,))
    n_pred = hat_next.shape[-2]
    if n_pred == 0:
        return jnp.asarray(0.0, dtype=hat_next.dtype)
    per = jnp.mean(jnp.sum((hat_next - target_next) ** 2, axis=-1), axis=-1)
    healthy = labels < 0.5
    count = jnp.sum(healthy.astype(hat_next.dtype))
    total = jnp.sum(jnp.where(healthy, per, 0.0))
    return jnp.where(count > 0.0, total / jnp.maximum(count, 1.0), 0.0)


def total_loss(outputs, y, lambda_seg: float = LAMBDA_SEG, lambda_norm: float = LAMBDA_NORM):
    """L_cls + lambda_seg * L_seg + lambda_norm * L_norm."""
    labels = jnp.asarray(y, dtype=jnp.float32).reshape((-1,))
    cls = classification_loss(outputs["sample_prob"], labels)
    seg = segment_loss(outputs["segment_risk"], labels)
    hat = outputs["hat_next"]
    target = outputs["tokens"][:, 1:, :]
    if hat.shape != target.shape:
        raise ValueError(
            f"hat_next shape {hat.shape} does not match next tokens {target.shape}"
        )
    norm = normative_loss(hat, target, labels)
    loss = cls + lambda_seg * seg + lambda_norm * norm
    return {"loss": loss, "cls": cls, "seg": seg, "norm": norm}
