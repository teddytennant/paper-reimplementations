"""CPU equation tests for TF, IDF, and normalized weights (eqs 3-6)."""

import math

import jax.numpy as jnp

from tfidf_ce.stats import (
    batch_document_frequency,
    smoothed_idf,
    term_frequency,
    tfidf_weights,
)


def test_idf_stopword_vs_hapax():
    """Stopword-like tokens with df=N have smaller idf than hapax tokens (eq 5)."""
    n = 16
    idf_stop = float(smoothed_idf(jnp.array([n], dtype=jnp.int32), n)[0])
    idf_hapax = float(smoothed_idf(jnp.array([1], dtype=jnp.int32), n)[0])
    assert math.isclose(idf_stop, math.log((1 + n) / (1 + n)) + 1, rel_tol=1e-6)
    assert math.isclose(idf_stop, 1.0, rel_tol=1e-6)
    want_hapax = math.log((1 + n) / 2) + 1
    assert math.isclose(idf_hapax, want_hapax, rel_tol=1e-6)
    assert idf_hapax > idf_stop


def test_term_frequency_repeat_vs_singleton():
    """A token that repeats 3 times has tf=3; a singleton has tf=1 (eq 3)."""
    # sequence: 7, 7, 7, 9, pad, pad
    tokens = jnp.array([[7, 7, 7, 9, 0, 0]], dtype=jnp.int32)
    tf = term_frequency(tokens, pad_id=0)
    assert float(tf[0, 0]) == 3.0
    assert float(tf[0, 1]) == 3.0
    assert float(tf[0, 2]) == 3.0
    assert float(tf[0, 3]) == 1.0
    assert float(tf[0, 4]) == 0.0
    assert float(tf[0, 5]) == 0.0


def test_padding_excluded_from_tf_of_real_tokens():
    """Pad tokens must not count toward tf of real tokens even if pad_id equals a count."""
    tokens = jnp.array([[2, 2, 0, 0]], dtype=jnp.int32)
    tf = term_frequency(tokens, pad_id=0)
    assert float(tf[0, 0]) == 2.0
    assert float(tf[0, 1]) == 2.0


def test_normalized_mean_weight_is_one():
    """After eq 6, mean(w) over non-pad tokens is 1; pad positions have w=0."""
    tokens = jnp.array(
        [
            [1, 1, 2, 0],
            [3, 4, 0, 0],
        ],
        dtype=jnp.int32,
    )
    vocab = 8
    df = batch_document_frequency(tokens, vocab, pad_id=0)
    n_docs = tokens.shape[0]
    w = tfidf_weights(tokens, df, n_docs, pad_id=0)
    valid = tokens != 0
    assert jnp.allclose(w.mean(where=valid), 1.0, atol=1e-5)
    assert jnp.all(w[~valid] == 0.0)


def test_padding_excluded_from_m():
    """If half the positions are pad, M is the non-pad count, not B*L."""
    tokens = jnp.array(
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
        ],
        dtype=jnp.int32,
    )
    b, length = tokens.shape
    assert b * length == 8
    m = int((tokens != 0).sum())
    assert m == 4
    df = batch_document_frequency(tokens, 8, pad_id=0)
    w = tfidf_weights(tokens, df, n_docs=2, pad_id=0)
    valid = tokens != 0
    # Mean over the 4 non-pad positions is 1; mean over all 8 is 0.5.
    assert jnp.allclose(w[valid].mean(), 1.0, atol=1e-5)
    assert jnp.allclose(w.mean(), 0.5, atol=1e-5)
    assert m != b * length


def test_rare_token_weight_higher_than_common():
    """Two identical sequences except a rare swap: that token's weight is higher.

    Each token appears once in its sequence (tf=1), so the difference is IDF:
    the common token is in every document, the rare token in one.
    """
    vocab = 32
    common = 1
    rare = 20
    tokens = jnp.array(
        [
            [common, 2, 3, 4],
            [common, 2, 3, 4],
            [common, 2, 3, 4],
            [common, 2, rare, 4],
        ],
        dtype=jnp.int32,
    )
    df = batch_document_frequency(tokens, vocab, pad_id=0)
    w = tfidf_weights(tokens, df, n_docs=4, pad_id=0)
    rare_w = float(w[3, 2])
    common_w = float(w[3, 0])
    assert float(term_frequency(tokens, pad_id=0)[3, 2]) == 1.0
    assert float(term_frequency(tokens, pad_id=0)[3, 0]) == 1.0
    assert rare_w > common_w
    assert int(df[common]) == 4
    assert int(df[rare]) == 1
