"""Rolling K-batch buffer for document-frequency estimates.

Section 3.1: IDF is estimated over a circular buffer of the last K mini-batches
(``N = B * K`` sequences when every batch has size B). Document frequency and
N include the current batch once it has been pushed. A token that falls out of
the oldest dropped batch leaves ``df``.
"""

from __future__ import annotations

from collections import deque

import jax.numpy as jnp
from jax import Array

from tfidf_ce.stats import batch_document_frequency, tfidf_weights


def update_df_from_batch(
    df: Array,
    tokens: Array,
    vocab_size: int,
    pad_id: int = 0,
    *,
    subtract: bool = False,
) -> Array:
    """Add (or subtract) per-sequence presence counts from a ``[V]`` df vector."""
    df = jnp.asarray(df, dtype=jnp.int32)
    delta = batch_document_frequency(tokens, vocab_size, pad_id)
    return df - delta if subtract else df + delta


class IdfBuffer:
    """Circular buffer of the last ``k`` token-id mini-batches.

    Constructor args ``k`` and ``vocab_size`` match the paper's buffer length
    and the size of the df vector. ``pad_id`` is excluded from df.

    Typical use (current batch included in df / N)::

        buf.push(tokens)
        weights = buf.weights(tokens)
    """

    def __init__(self, k: int, vocab_size: int, pad_id: int = 0):
        if k < 1:
            raise ValueError("k must be >= 1")
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.k = int(k)
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self._slots: deque[Array] = deque(maxlen=self.k)

    def push(self, tokens: Array) -> IdfBuffer:
        """Insert a ``[B, L]`` batch. Drops the oldest once ``k`` slots are full."""
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [B, L], got shape {tokens.shape}")
        self._slots.append(tokens)
        return self

    @property
    def n_docs(self) -> int:
        """Number of sequences currently in the buffer (``N``)."""
        return int(sum(int(batch.shape[0]) for batch in self._slots))

    @property
    def df(self) -> Array:
        """Document frequency ``[V]`` over sequences currently in the buffer."""
        acc = jnp.zeros((self.vocab_size,), dtype=jnp.int32)
        for batch in self._slots:
            acc = update_df_from_batch(
                acc, batch, self.vocab_size, self.pad_id, subtract=False
            )
        return acc

    def weights(self, tokens: Array) -> Array:
        """TF-IDF weights for ``tokens`` using df/N of the current buffer.

        Push the batch first if it should be included in df/N.
        """
        if self.n_docs == 0:
            raise ValueError("buffer is empty; push a batch before computing weights")
        return tfidf_weights(tokens, self.df, self.n_docs, pad_id=self.pad_id)

    def update(self, tokens: Array) -> Array:
        """Push ``tokens`` (so df/N include it) and return its TF-IDF weights."""
        self.push(tokens)
        return self.weights(tokens)
