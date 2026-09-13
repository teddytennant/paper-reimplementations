"""Weighted CE vs uniform CE, scalar loss, pad-zero gradients."""

import jax
import jax.numpy as jnp

from tfidf_ce.loss import mean_cross_entropy, tfidf_cross_entropy
from tfidf_ce.stats import batch_document_frequency, tfidf_weights


def _one_hot_logits(tokens, vocab, pad_id=0, peak=10.0):
    """Logits that peak on the true token (except pads)."""
    b, length = tokens.shape
    logits = jnp.zeros((b, length, vocab), dtype=jnp.float32)
    idx_b = jnp.arange(b)[:, None]
    idx_t = jnp.arange(length)[None, :]
    logits = logits.at[idx_b, idx_t, tokens].set(peak)
    return logits


def test_weighted_ce_with_unit_weights_matches_mean_ce():
    vocab = 8
    tokens = jnp.array(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=jnp.int32,
    )
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (2, 4, vocab))
    valid = tokens != 0
    weights = valid.astype(jnp.float32)
    weighted = tfidf_cross_entropy(logits, tokens, weights, pad_id=0)
    plain = mean_cross_entropy(logits, tokens, pad_id=0)
    assert jnp.allclose(weighted, plain, atol=1e-5)
    assert weighted.shape == ()


def test_loss_is_scalar_and_eq2_when_w_is_one():
    vocab = 6
    tokens = jnp.array([[1, 2, 3], [2, 1, 3]], dtype=jnp.int32)
    logits = _one_hot_logits(tokens, vocab)
    weights = jnp.ones_like(tokens, dtype=jnp.float32)
    loss = tfidf_cross_entropy(logits, tokens, weights, pad_id=0)
    eq2 = mean_cross_entropy(logits, tokens, pad_id=0)
    assert loss.shape == ()
    assert jnp.allclose(loss, eq2, atol=1e-5)


def test_pad_positions_contribute_zero_gradient():
    vocab = 8
    tokens = jnp.array([[1, 2, 0, 0]], dtype=jnp.int32)
    key = jax.random.PRNGKey(1)
    logits = jax.random.normal(key, (1, 4, vocab))
    weights = (tokens != 0).astype(jnp.float32)

    def loss_fn(lg):
        return tfidf_cross_entropy(lg, tokens, weights, pad_id=0)

    grads = jax.grad(loss_fn)(logits)
    assert grads.shape == logits.shape
    assert jnp.allclose(grads[0, 2], 0.0, atol=1e-6)
    assert jnp.allclose(grads[0, 3], 0.0, atol=1e-6)
    assert jnp.any(jnp.abs(grads[0, 0]) > 0)
    assert jnp.any(jnp.abs(grads[0, 1]) > 0)


def test_tfidf_loss_uses_normalized_weights():
    vocab = 16
    tokens = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 5, 6, 7],
        ],
        dtype=jnp.int32,
    )
    df = batch_document_frequency(tokens, vocab, pad_id=0)
    w = tfidf_weights(tokens, df, n_docs=2, pad_id=0)
    key = jax.random.PRNGKey(2)
    logits = jax.random.normal(key, (2, 4, vocab))
    loss = tfidf_cross_entropy(logits, tokens, w, pad_id=0)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    # Token 1 is in both sequences; token 7 is a hapax. Same tf=1, so IDF wins.
    assert float(w[0, 0]) < float(w[1, 3])
