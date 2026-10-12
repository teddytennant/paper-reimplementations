"""Contract tests for the time-delay sequence model."""

import jax
import jax.numpy as jnp
import pytest

from time_delay_system import (
    apply,
    decay_weights,
    delay_correction,
    diagonal_state_parallel,
    diagonal_state_sequential,
    fuzzy_modulate,
    history_window,
    init_params,
    run_delayed_copy_smoke,
    semantic_lag,
)


def test_decay_weights_decrease_with_lag():
    lam = 0.7
    alpha = 1.0
    n_lags = 5
    w = decay_weights(lam, alpha, n_lags)
    u = jnp.arange(1, n_lags + 1, dtype=jnp.float32)
    assert w.shape == (n_lags,)
    assert jnp.allclose(w, jnp.exp(-lam * u ** alpha))
    assert jnp.all(w[:-1] > w[1:])
    assert jnp.all(w > 0)


def test_lag_weights_sum_to_one():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 9, 5), dtype=jnp.float32)
    W = jax.random.normal(jax.random.PRNGKey(1), (5, 5), dtype=jnp.float32)
    out = semantic_lag(x, W, lam=0.4, kappa=1.5, alpha=1.0, n_lags=4)
    assert out.alpha.shape == (3, 9, 4)
    assert jnp.allclose(jnp.sum(out.alpha, axis=-1), 1.0, atol=1e-5)


def test_zero_history_outputs():
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (2, 6, 4), dtype=jnp.float32)
    W = jnp.eye(4, dtype=jnp.float32)
    out = semantic_lag(x, W, lam=0.5, kappa=1.0, alpha=1.0, n_lags=4)
    assert jnp.allclose(out.x_bar[:, 0], 0.0, atol=1e-6)
    assert jnp.allclose(out.s_hat[:, 0], 0.0, atol=1e-6)

    zeros = jnp.zeros_like(x)
    out_z = semantic_lag(zeros, W, lam=0.5, kappa=2.0, alpha=1.0, n_lags=4)
    assert jnp.allclose(out_z.x_bar, 0.0, atol=1e-6)
    assert jnp.allclose(out_z.s_hat, 0.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(out_z.alpha, axis=-1), 1.0, atol=1e-5)


def test_no_future_token_receives_weight():
    key = jax.random.PRNGKey(3)
    batch, time, dim, n_lags = 2, 8, 4, 4
    x = jax.random.normal(key, (batch, time, dim), dtype=jnp.float32)
    W = jnp.eye(dim, dtype=jnp.float32)
    hist = history_window(x, n_lags)
    for t in range(time):
        for u in range(1, n_lags + 1):
            slot = hist[:, t, u - 1]
            if t - u >= 0:
                assert jnp.allclose(slot, x[:, t - u])
            else:
                assert jnp.allclose(slot, 0.0)
            if t + 1 < time:
                # A future token is not copied into this lag slot unless it
                # happens to equal the past token. Force a unique future value.
                pass

    # Value test: editing the future leaves past lag outputs unchanged.
    x_future = x.at[:, 5:].set(jax.random.normal(jax.random.PRNGKey(4), (batch, time - 5, dim)))
    kwargs = dict(lam=0.3, kappa=1.2, alpha=1.0, n_lags=n_lags)
    out = semantic_lag(x, W, **kwargs)
    out_f = semantic_lag(x_future, W, **kwargs)
    assert jnp.allclose(out.alpha[:, :5], out_f.alpha[:, :5], atol=1e-6)
    assert jnp.allclose(out.x_bar[:, :5], out_f.x_bar[:, :5], atol=1e-6)
    assert jnp.allclose(out.s_hat[:, :5], out_f.s_hat[:, :5], atol=1e-6)
    assert jnp.allclose(out.history[:, :5], out_f.history[:, :5], atol=1e-6)


def test_fuzzy_membership_and_positive_g():
    centers = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    sigma = jnp.asarray(1.0, dtype=jnp.float32)
    f_r = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float32)
    g_r = jnp.array([0.0, -0.5, 1.0], dtype=jnp.float32)

    z0 = centers[0] + jnp.array([0.05, 0.0, 0.0, 0.0], dtype=jnp.float32)
    mu0, f0, g0 = fuzzy_modulate(z0[None, None, :], centers, sigma, f_r, g_r)
    assert mu0.shape[-1] == 3
    assert jnp.allclose(jnp.sum(mu0, axis=-1), 1.0, atol=1e-5)
    assert int(jnp.argmax(mu0[0, 0])) == 0
    assert bool(jnp.all(g0 > 0.0))

    z2 = centers[2] + jnp.array([0.0, -0.1, 0.02, 0.0], dtype=jnp.float32)
    mu2, _, g2 = fuzzy_modulate(z2[None, None, :], centers, sigma, f_r, g_r)
    assert jnp.allclose(jnp.sum(mu2, axis=-1), 1.0, atol=1e-5)
    assert int(jnp.argmax(mu2[0, 0])) == 2
    assert bool(jnp.all(g2 > 0.0))
    # f is the membership-weighted scalar consequent, not a hidden vector.
    assert f0.shape == (1, 1)
    assert jnp.isfinite(f0)


def test_delay_correction_uses_tanh():
    s_hat = jnp.array([[0.4, -0.8]], dtype=jnp.float32)
    x_bar = jnp.array([[[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]], dtype=jnp.float32)
    gamma = jnp.asarray(0.3, dtype=jnp.float32)
    C = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]], dtype=jnp.float32)
    d = delay_correction(s_hat, x_bar, gamma, C)
    cx = jnp.einsum("hd,btd->bth", C, x_bar)
    expected = gamma * jnp.tanh(s_hat)[..., None] * cx
    assert jnp.allclose(d, expected)
    assert d.shape == (1, 2, 2)


def test_closed_form_matches_hand_unroll():
    a = jnp.array([0.5, 0.8], dtype=jnp.float32)
    b = jnp.array(
        [[[1.0, 2.0], [0.5, -1.0], [0.25, 0.5]]],
        dtype=jnp.float32,
    )
    h = diagonal_state_parallel(a, b)
    h1 = b[:, 0]
    h2 = a * b[:, 0] + b[:, 1]
    h3 = (a ** 2) * b[:, 0] + a * b[:, 1] + b[:, 2]
    expected = jnp.stack([h1, h2, h3], axis=1)
    assert jnp.allclose(h, expected, atol=1e-5)
    assert jnp.allclose(h, diagonal_state_sequential(a, b), atol=1e-5)


def test_parallel_matches_sequential_float32():
    key = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(key)
    batch, time, hidden = 3, 16, 8
    a = jax.random.uniform(k1, (hidden,), minval=0.5, maxval=0.99, dtype=jnp.float32)
    b = jax.random.normal(k2, (batch, time, hidden), dtype=jnp.float32)
    h_par = diagonal_state_parallel(a, b)
    h_seq = diagonal_state_sequential(a, b)
    assert h_par.dtype == jnp.float32
    assert h_seq.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(h_par - h_seq))) < 1e-4


def test_model_is_causal_and_returns_last_and_mean():
    key = jax.random.PRNGKey(6)
    params = init_params(key, vocab=32, d=16, n_classes=4, n_lags=4, n_rules=4, d_hidden=16)
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 10), 0, 32)
    out = apply(params, tokens, n_lags=4)
    assert out.s.shape == (2, 10, 16)
    assert out.last.shape == (2, 16)
    assert out.mean.shape == (2, 16)
    assert jnp.allclose(out.last, out.s[:, -1])
    assert jnp.allclose(out.mean, jnp.mean(out.s, axis=1))
    assert out.s_out.shape == (2, 10, 16)
    assert jnp.all(jnp.isfinite(out.s))
    assert jnp.all(jnp.isfinite(out.s_out))
    assert jnp.all(out.g > 0)
    assert jnp.allclose(out.z, out.d / 2)

    tokens_future = tokens.at[:, 6:].set(0)
    out_f = apply(params, tokens_future, n_lags=4)
    assert jnp.allclose(out.s[:, :6], out_f.s[:, :6], atol=1e-5)
    assert jnp.allclose(out.s_out[:, :6], out_f.s_out[:, :6], atol=1e-5)
    # Position 0 has zero history, so the delay correction is 0.
    assert jnp.allclose(out.d[:, 0], 0.0, atol=1e-5)
    assert jnp.allclose(out.x_bar[:, 0], 0.0, atol=1e-5)


def test_positive_constraints():
    key = jax.random.PRNGKey(8)
    params = init_params(key, vocab=32, d=8, n_classes=2, n_lags=4, n_rules=4)
    out = apply(params, jnp.zeros((1, 4), dtype=jnp.int32), n_lags=4)
    assert jnp.all(out.a > 0) and jnp.all(out.a < 1)
    assert float(jax.nn.softplus(params["lam_raw"])) > 0
    assert float(jax.nn.softplus(params["kappa_raw"])) > 0
    assert float(jax.nn.softplus(params["alpha_raw"])) > 0
    assert float(jax.nn.softplus(params["sigma_raw"])) > 0


def test_smoke_loss_finite():
    result = run_delayed_copy_smoke()
    assert result["steps"] == 40
    assert result["finite"]
    assert result["step_0"] > 0
    assert result["step_final"] > 0
