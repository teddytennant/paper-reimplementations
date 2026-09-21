"""Q-constrained concept residual attention (eqs. 14–21)."""

import jax.numpy as jnp
import numpy as np

from aracd.ops import layer_norm
from aracd.residual import (
    attention_query,
    attention_scores,
    concept_residual,
    q_masked_softmax,
)


def _numpy_layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def test_query_eq15_matches_numpy_ln_tanh():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(5, 8)).astype(np.float32)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    got = np.asarray(attention_query(jnp.asarray(e), jnp.asarray(v), eps=1e-5))
    ref = _numpy_layer_norm(np.tanh(e + v), eps=1e-5)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    # last-axis population LN: mean ~ 0, std ~ 1
    np.testing.assert_allclose(got.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(got.std(axis=-1), 1.0, atol=1e-4)
    ln_direct = np.asarray(layer_norm(jnp.tanh(jnp.asarray(e) + jnp.asarray(v))))
    np.testing.assert_allclose(got, ln_direct, rtol=1e-6, atol=1e-6)


def test_scores_eq17_scaled_dot():
    q = jnp.array([0.2, -0.4, 0.1, 0.8], dtype=jnp.float32)
    k = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    scores = attention_scores(q, k)
    d_att = 4
    expected = (np.asarray(q) @ np.asarray(k).T) / np.sqrt(d_att)
    np.testing.assert_allclose(np.asarray(scores), expected, rtol=1e-6, atol=1e-6)


def test_q_mask_eq18_19_zero_off_support_and_sum_one():
    scores = jnp.array([2.0, -1.0, 0.5, 3.0], dtype=jnp.float32)
    q_i = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    alpha = np.asarray(q_masked_softmax(scores, q_i))
    np.testing.assert_allclose(alpha[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(alpha[3], 0.0, atol=1e-6)
    np.testing.assert_allclose(alpha.sum(), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(alpha[q_i.astype(bool)].sum(), 1.0, rtol=1e-6)
    # unmasked softmax on the support
    support = np.array([2.0, 0.5], dtype=np.float32)
    e = np.exp(support - support.max())
    ref = e / e.sum()
    np.testing.assert_allclose(alpha[[0, 2]], ref, rtol=1e-5, atol=1e-6)


def test_single_concept_item_alpha_one_and_r_att():
    scores = jnp.array([0.3, 4.0, -2.0], dtype=jnp.float32)
    q_i = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    alpha = q_masked_softmax(scores, q_i)
    np.testing.assert_allclose(np.asarray(alpha), np.array([0.0, 1.0, 0.0]), atol=1e-6)
    b_k = jnp.array([0.1, -0.4, 0.7], dtype=jnp.float32)
    b_s_k = jnp.array([0.2, 0.5, -0.1], dtype=jnp.float32)
    r = concept_residual(alpha, b_k, b_s_k)
    np.testing.assert_allclose(np.asarray(r), float(b_k[1] + b_s_k[1]), rtol=1e-6, atol=1e-6)


def test_concept_residual_eq14_weighted_sum():
    alpha = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)
    b_k = jnp.array([1.0, -1.0, 0.5], dtype=jnp.float32)
    b_s_k = jnp.array([0.0, 0.4, -0.2], dtype=jnp.float32)
    r = concept_residual(alpha, b_k, b_s_k)
    values = np.asarray(b_k) + np.asarray(b_s_k)
    expected = float(np.dot(np.asarray(alpha), values))
    np.testing.assert_allclose(np.asarray(r), expected, rtol=1e-6, atol=1e-6)
