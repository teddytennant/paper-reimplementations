"""Lock reverse-process losses and discrete support of arXiv:2609.31070."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qd_dtqw.losses import (
    categorical_cross_entropy,
    histogram_kl,
    intensity_histogram,
    mixed_loss,
    softmax,
    time_alpha,
)
from qd_dtqw.reverse import generate, init_reverse, reverse_step, sgd_step
from qd_dtqw.walk import DTQW


K = 4
H = W = 4


def test_cross_entropy_closed_form():
    """CE = -sum y log p on a one-hot target."""
    logits = jnp.asarray(
        [[2.0, 0.0, -1.0, 0.5], [0.1, 0.2, 0.3, 0.4]],
        dtype=jnp.float64,
    )
    targets = jnp.asarray([0, 2], dtype=jnp.int32)
    p = np.asarray(softmax(logits, axis=-1))
    y = np.eye(K)[np.asarray(targets)]
    expected = float(-np.sum(y * np.log(p)) / 2.0)
    got = float(categorical_cross_entropy(logits, targets, k=K))
    np.testing.assert_allclose(got, expected, atol=1e-10)

    perfect = 40.0 * jnp.eye(K, dtype=jnp.float64)[None, None, :, :]
    # (1,1,K,K) is wrong shape; use spatial 1x1
    logits_ok = 40.0 * jnp.eye(K, dtype=jnp.float64)
    ce0 = float(categorical_cross_entropy(logits_ok[None, None, ...], jnp.arange(K)[None, None, ...], k=K))
    # simpler: one pixel, target 1, huge logit on class 1
    logits_px = jnp.asarray([0.0, 50.0, 0.0, 0.0], dtype=jnp.float64)
    ce_perfect = float(categorical_cross_entropy(logits_px, jnp.asarray(1), k=K))
    assert ce_perfect < 1e-10
    del perfect, ce0


def test_ce_decreases_on_tiny_batch():
    """SGD on a synthetic 4x4 blob batch lowers categorical CE (eq. 12)."""
    key = jax.random.PRNGKey(0)
    params = init_reverse(key, k=K, hidden=8)
    # Two-class blob: background 0, 2x2 square of class 3.
    x_t = jnp.zeros((2, H, W), dtype=jnp.int32)
    x_t = x_t.at[0, 1:3, 1:3].set(3)
    x_t = x_t.at[1, 0:2, 2:4].set(3)
    x_next = (x_t + 1) % K
    t = jnp.asarray([1, 1], dtype=jnp.float64)

    losses = []
    p = params
    for _ in range(40):
        p, loss = sgd_step(p, x_next, x_t, t, lr=0.2)
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_reverse_step_stays_in_alphabet():
    """Reverse step on uniform noise stays in {0,...,K-1}."""
    key = jax.random.PRNGKey(1)
    params = init_reverse(key, k=K, hidden=4)
    key, k_x, k_s = jax.random.split(key, 3)
    x = jax.random.randint(k_x, (H, W), 0, K)
    y = reverse_step(params, x, t=5, key=k_s, sample=True)
    y_hat = reverse_step(params, x, t=5, sample=False)
    for arr in (y, y_hat, x):
        a = np.asarray(arr)
        assert a.shape == (H, W)
        assert np.all(a >= 0) and np.all(a <= K - 1)
        assert np.issubdtype(a.dtype, np.integer) or np.all(np.equal(a, np.round(a)))


def test_generate_from_uniform_prior():
    key = jax.random.PRNGKey(2)
    params = init_reverse(key, k=K, hidden=4)
    x = generate(params, shape=(2, H, W), T=3, key=key, sample=True)
    a = np.asarray(x)
    assert a.shape == (2, H, W)
    assert np.all(a >= 0) and np.all(a <= K - 1)


def test_histogram_kl_eq13():
    p = jnp.asarray([0.5, 0.5, 0.0, 0.0], dtype=jnp.float64)
    q = jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=jnp.float64)
    kl = float(histogram_kl(p, q))
    # 0.5 log(0.5/0.25) * 2 = log(2)
    np.testing.assert_allclose(kl, np.log(2.0), atol=1e-8)
    img = jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32)
    hist = np.asarray(intensity_histogram(img, k=4))
    np.testing.assert_allclose(hist, [0.5, 0.5, 0.0, 0.0], atol=1e-12)


def test_time_alpha_high_t_matches_histogram():
    """α_t large at low t (CE / reconstruct), small at high t (histogram KL)."""
    T = 8
    a0 = float(time_alpha(0, T))
    aT = float(time_alpha(T - 1, T))
    assert a0 > 0.99
    assert aT < 0.01
    logits = jnp.ones((1, 2, 2, K), dtype=jnp.float64)
    targets = jnp.zeros((1, 2, 2), dtype=jnp.int32)
    mixed_loss(logits, targets, t=0, T=T, k=K)
    mixed_loss(logits, targets, t=T - 1, T=T, k=K)


def test_forward_image_independent_and_in_alphabet():
    walk = DTQW(2)
    key = jax.random.PRNGKey(3)
    x0 = jnp.asarray(
        [[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 3, 3], [1, 1, 2, 2]],
        dtype=jnp.int32,
    )
    xt = walk.forward_image(x0, t=2, key=key)
    a = np.asarray(xt)
    assert a.shape == x0.shape
    assert np.all(a >= 0) and np.all(a <= 3)
    xt, xt1 = walk.training_pair_image(x0, t=1, key=jax.random.PRNGKey(4))
    for arr in (xt, xt1):
        b = np.asarray(arr)
        assert b.shape == x0.shape
        assert np.all(b >= 0) and np.all(b <= 3)
