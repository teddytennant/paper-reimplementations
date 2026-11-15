"""TF-IDF weighted cross-entropy (eq 1), with uniform CE as the w_i = 1 case (eq 2).

Logits are aligned with tokens: ``logits[b, i]`` is the distribution for
``tokens[b, i]``. For next-token language modeling, shift before the call::

    loss = tfidf_cross_entropy(logits[:, :-1], tokens[:, 1:], weights[:, 1:], pad_id)

The mean is over the M non-padding positions, not over padded length L. With
no padding this matches the paper's ``1/L`` factor. Pad positions are masked
out of the mean, so they contribute 0 to the value and to gradients w.r.t. logits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def token_nll(logits: Array, tokens: Array) -> Array:
    """Per-position negative log-likelihood, shape ``[B, L]``."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, tokens[..., None], axis=-1).squeeze(-1)


def mean_cross_entropy(logits: Array, tokens: Array, pad_id: int = 0) -> Array:
    """Unweighted mean NLL over non-padding positions (eq 2 with pad excluded)."""
    nll = token_nll(logits, tokens)
    mask = jnp.asarray(tokens, dtype=jnp.int32) != pad_id
    m = mask.sum().astype(logits.dtype)
    return (nll * mask).sum() / jnp.maximum(m, 1)


def tfidf_cross_entropy(
    logits: Array,
    tokens: Array,
    weights: Array,
    pad_id: int = 0,
) -> Array:
    """Weighted mean NLL over non-padding positions (eq 1).

    ``logits`` has shape ``[B, L, V]`` (or any leading dims + vocab), aligned
    with ``tokens`` and ``weights`` of shape ``[B, L]``. Returns a scalar.
    When ``weights`` is 1 on every non-pad position, this equals
    ``mean_cross_entropy``.
    """
    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    weights = jnp.asarray(weights, dtype=logits.dtype)
    nll = token_nll(logits, tokens)
    mask = tokens != pad_id
    m = mask.sum().astype(logits.dtype)
    return (nll * weights * mask).sum() / jnp.maximum(m, 1)
