"""Rolling K-batch document-frequency buffer."""

import jax.numpy as jnp

from tfidf_ce.buffer import IdfBuffer, update_df_from_batch
from tfidf_ce.stats import batch_document_frequency


def test_buffer_caps_at_k_batches():
    """After feeding K+2 batches, N = B*K (not more)."""
    k = 3
    vocab = 32
    pad_id = 0
    b = 2
    length = 4
    buf = IdfBuffer(k=k, vocab_size=vocab, pad_id=pad_id)
    batches = []
    for i in range(k + 2):
        # Distinct filler token per batch so we can drop the oldest.
        batch = jnp.full((b, length), i + 1, dtype=jnp.int32)
        batches.append(batch)
        buf.push(batch)
    assert buf.n_docs == b * k
    assert len(buf._slots) == k


def test_df_equals_n_when_token_in_every_buffered_sequence():
    k = 3
    vocab = 32
    b = 2
    length = 4
    stop = 5
    buf = IdfBuffer(k=k, vocab_size=vocab, pad_id=0)
    for i in range(k + 2):
        batch = jnp.full((b, length), stop, dtype=jnp.int32)
        batch = batch.at[:, -1].set(10 + i)
        buf.push(batch)
    assert buf.n_docs == b * k
    assert int(buf.df[stop]) == buf.n_docs


def test_dropped_batch_leaves_df():
    """A token only in the oldest dropped batch is gone from df."""
    k = 2
    vocab = 32
    buf = IdfBuffer(k=k, vocab_size=vocab, pad_id=0)
    only_old = 11
    later = 12
    batch0 = jnp.array([[only_old, only_old], [only_old, only_old]], dtype=jnp.int32)
    batch1 = jnp.array([[later, later], [later, later]], dtype=jnp.int32)
    batch2 = jnp.array([[later, later], [later, later]], dtype=jnp.int32)
    buf.push(batch0)
    assert int(buf.df[only_old]) == 2
    buf.push(batch1)
    assert int(buf.df[only_old]) == 2
    buf.push(batch2)  # drops batch0
    assert buf.n_docs == 4
    assert int(buf.df[only_old]) == 0
    assert int(buf.df[later]) == 4


def test_update_df_from_batch_add_and_subtract():
    vocab = 8
    tokens = jnp.array([[1, 2, 0], [1, 3, 0]], dtype=jnp.int32)
    df = jnp.zeros((vocab,), dtype=jnp.int32)
    df = update_df_from_batch(df, tokens, vocab, pad_id=0)
    assert int(df[1]) == 2
    assert int(df[2]) == 1
    assert int(df[0]) == 0
    df = update_df_from_batch(df, tokens, vocab, pad_id=0, subtract=True)
    assert jnp.all(df == 0)


def test_weights_include_current_batch():
    """Push then weights: df/N include the current batch."""
    vocab = 16
    buf = IdfBuffer(k=4, vocab_size=vocab, pad_id=0)
    tokens = jnp.array([[1, 2, 3], [1, 1, 4]], dtype=jnp.int32)
    w = buf.update(tokens)
    assert buf.n_docs == 2
    assert int(buf.df[1]) == 2
    valid = tokens != 0
    assert jnp.allclose(w[valid].mean(), 1.0, atol=1e-5)
    # Same as computing df from this batch alone.
    df = batch_document_frequency(tokens, vocab, pad_id=0)
    assert jnp.array_equal(buf.df, df)
