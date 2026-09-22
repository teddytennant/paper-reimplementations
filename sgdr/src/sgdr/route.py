"""Causal top-p block mask (eq 13 in the paper's selection step).

For each query block i, softmax is taken over causal key blocks j <= i
after a max shift. The smallest prefix of blocks in descending score order
whose softmax mass is at least p is kept. The diagonal block j = i is
always on. p = 1 selects every causal block.

If the diagonal score is below the minimal prefix, it is still selected.
In that case the selected set is not a pure score prefix, so dropping the
lowest selected block need not push the mass under p. The mass of the
score prefix itself does.
"""

import numpy as np
import jax.numpy as jnp


def top_p_mask(scores, p=0.9):
    """Return a boolean mask of shape (n_q_blocks, n_k_blocks)."""
    scores = jnp.asarray(scores, dtype=jnp.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be rank 2, got {scores.shape}")
    n_q, n_k = scores.shape
    host = np.asarray(scores)
    mask = np.zeros((n_q, n_k), dtype=bool)
    threshold = float(p)
    for i in range(n_q):
        causal_n = min(i + 1, n_k)
        if causal_n == 0:
            continue
        row = host[i, :causal_n]
        if threshold >= 1.0:
            mask[i, :causal_n] = True
            continue
        shifted = row - row.max()
        weights = np.exp(shifted)
        weights = weights / weights.sum()
        order = np.argsort(-row, kind="stable")
        cumulative = np.cumsum(weights[order])
        keep = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        keep = min(keep, causal_n)
        mask[i, order[:keep]] = True
        if i < n_k:
            mask[i, i] = True
    return jnp.asarray(mask)


def causal_softmax(scores):
    """Max-shifted softmax over j <= i. Future blocks get weight 0.

    Returns probabilities of shape scores.
    """
    scores = jnp.asarray(scores, dtype=jnp.float64)
    n_q, n_k = scores.shape
    causal = jnp.arange(n_k)[None, :] <= jnp.arange(n_q)[:, None]
    masked = jnp.where(causal, scores, -jnp.inf)
    row_max = jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True)
    shifted = jnp.where(causal, scores - row_max, -jnp.inf)
    weights = jnp.exp(shifted)
    weights = jnp.where(causal, weights, 0.0)
    denom = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.maximum(denom, 1e-30)
