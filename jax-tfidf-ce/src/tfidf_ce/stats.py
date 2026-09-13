"""TF, DF, smoothed IDF, and batch-normalized TF-IDF weights.

Paper (arXiv:2609.11029) equations 3-6. Padding tokens are excluded from
term counts, document frequency, the non-pad count M, and the loss.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _as_tokens(tokens: Array) -> Array:
    return jnp.asarray(tokens, dtype=jnp.int32)


def term_frequency(tokens: Array, pad_id: int = 0) -> Array:
    """Within-sequence term frequency (eq 3).

    ``tf[b, i]`` is the number of non-padding positions in sequence ``b``
    whose token equals ``tokens[b, i]``. Pad positions are 0 and do not
    contribute to the count of any real token.
    """
    tokens = _as_tokens(tokens)
    valid = tokens != pad_id
    same = tokens[:, :, None] == tokens[:, None, :]
    counted = same & valid[:, None, :]
    tf = counted.sum(axis=-1).astype(jnp.float32)
    return jnp.where(valid, tf, 0.0)


def sequence_presence(tokens: Array, vocab_size: int, pad_id: int = 0) -> Array:
    """Indicator ``[B, V]``: token ``v`` occurs in sequence ``b`` (pad ignored)."""
    tokens = _as_tokens(tokens)
    valid = (tokens != pad_id)[..., None]
    one_hot = (jnp.arange(vocab_size, dtype=jnp.int32) == tokens[..., None]) & valid
    return one_hot.any(axis=1).astype(jnp.int32)


def batch_document_frequency(
    tokens: Array, vocab_size: int, pad_id: int = 0
) -> Array:
    """``df[v]`` = number of sequences in the batch that contain ``v`` (eq 4)."""
    return sequence_presence(tokens, vocab_size, pad_id).sum(axis=0).astype(jnp.int32)


def smoothed_idf(df: Array, n_docs: int | Array) -> Array:
    """Smoothed inverse document frequency (eq 5).

    ``idf(v) = log((1 + N) / (1 + df(v))) + 1`` with natural log.
    """
    df_f = jnp.asarray(df, dtype=jnp.float32)
    n = jnp.asarray(n_docs, dtype=jnp.float32)
    return jnp.log((1.0 + n) / (1.0 + df_f)) + 1.0


def tfidf_weights(
    tokens: Array,
    df: Array,
    n_docs: int | Array,
    pad_id: int = 0,
) -> Array:
    """Normalized TF-IDF token weights for one mini-batch (eq 6).

    ``tokens`` is ``[B, L]`` int32. ``df`` is a length-``V`` document-frequency
    vector (typically from the rolling buffer, including this batch). Raw
    weights ``w' = tf * idf(x_i)`` are divided by their mean over the ``M``
    non-padding tokens in *this* batch so ``E[w] = 1`` on those positions.
    Pad positions receive weight 0.
    """
    tokens = _as_tokens(tokens)
    valid = tokens != pad_id
    tf = term_frequency(tokens, pad_id=pad_id)
    idf = smoothed_idf(df, n_docs)
    raw = tf * idf[tokens]
    raw = jnp.where(valid, raw, 0.0)
    m = valid.sum().astype(jnp.float32)
    mean_raw = jnp.where(m > 0, raw.sum() / jnp.maximum(m, 1.0), 1.0)
    return jnp.where(valid, raw / mean_raw, 0.0)
