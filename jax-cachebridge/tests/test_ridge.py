import jax
import jax.numpy as jnp
import pytest

from cachebridge.ridge import (apply_affine, fit_affine, ridge_objective,
                               sufficient_statistics, weighted_mean)


def _data(key, n=90, p=12, d=5, seed_scale=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (n, p)) * seed_scale
    Y = jax.random.normal(k2, (n, d))
    w = jax.random.uniform(k3, (n,), minval=0.2, maxval=2.5)
    return X, Y, n * w / w.sum()          # mean-one weights, eq 8


def test_matches_dense_weighted_least_squares():
    """Eq 10 against an augmented sqrt-weight lstsq, a different numerical route."""
    X, Y, w = _data(jax.random.PRNGKey(0))
    lam = 0.03
    W, b = fit_affine(X, Y, w, lam)

    xbar, ybar = weighted_mean(X, w), weighted_mean(Y, w)
    sw = jnp.sqrt(w)[:, None]
    design = jnp.concatenate([sw * (X - xbar), jnp.sqrt(lam) * jnp.eye(X.shape[1])])
    rhs = jnp.concatenate([sw * (Y - ybar), jnp.zeros((X.shape[1], Y.shape[1]))])
    W_ref = jnp.linalg.lstsq(design, rhs, rcond=None)[0]

    assert jnp.max(jnp.abs(W - W_ref)) < 1e-9
    assert jnp.max(jnp.abs(b - (ybar - xbar @ W_ref))) < 1e-9


def test_gradient_of_eq8_objective_vanishes():
    """The eq 8 objective is stationary in both W and b at the eq 10 solution."""
    X, Y, w = _data(jax.random.PRNGKey(1))
    lam = 0.05
    W, b = fit_affine(X, Y, w, lam)
    gW, gb = jax.grad(ridge_objective, argnums=(0, 1))(W, b, X, Y, w, lam)
    scale = jnp.sum(w * jnp.sum(Y**2, -1))
    assert jnp.max(jnp.abs(gW)) / scale < 1e-12
    assert jnp.max(jnp.abs(gb)) / scale < 1e-12


def test_perturbing_the_solution_raises_the_objective():
    X, Y, w = _data(jax.random.PRNGKey(2))
    lam = 0.05
    W, b = fit_affine(X, Y, w, lam)
    base = ridge_objective(W, b, X, Y, w, lam)
    noise = jax.random.normal(jax.random.PRNGKey(9), W.shape)
    for step in (1e-3, 1e-2, 1e-1):
        assert ridge_objective(W + step * noise, b, X, Y, w, lam) > base


def test_vanishing_lambda_gives_ordinary_least_squares():
    X, Y, _ = _data(jax.random.PRNGKey(3), n=120, p=10)
    w = jnp.ones(X.shape[0])
    W, b = fit_affine(X, Y, w, lam=1e-12)
    aug = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    ols = jnp.linalg.lstsq(aug, Y, rcond=None)[0]
    assert jnp.max(jnp.abs(W - ols[:-1])) < 1e-7
    assert jnp.max(jnp.abs(b - ols[-1])) < 1e-7


def test_recovers_a_planted_affine_map_without_noise():
    key = jax.random.PRNGKey(4)
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (200, 8))
    W_true = jax.random.normal(k2, (8, 4))
    b_true = jax.random.normal(k3, (4,))
    Y = X @ W_true + b_true
    W, b = fit_affine(X, Y, None, lam=1e-11)
    assert jnp.max(jnp.abs(W - W_true)) < 1e-8
    assert jnp.max(jnp.abs(b - b_true)) < 1e-8
    assert jnp.max(jnp.abs(apply_affine(X, W, b) - Y)) < 1e-8


@pytest.mark.parametrize("lam", [1e-4, 0.01, 1.0])
def test_ridge_shrinks_monotonically(lam):
    X, Y, w = _data(jax.random.PRNGKey(5))
    W, _ = fit_affine(X, Y, w, lam)
    W_small, _ = fit_affine(X, Y, w, lam / 10.0)
    assert jnp.sum(W**2) < jnp.sum(W_small**2)


def test_head_batched_stats_match_per_head_loop():
    key = jax.random.PRNGKey(6)
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (3, 40, 6))
    Y = jax.random.normal(k2, (3, 40, 2))
    w = jax.random.uniform(k3, (3, 40), minval=0.5, maxval=1.5)
    w = 40 * w / w.sum(-1, keepdims=True)
    A, B, _, _ = sufficient_statistics(X, Y, w)
    for h in range(3):
        Ah, Bh, _, _ = sufficient_statistics(X[h], Y[h], w[h])
        assert jnp.max(jnp.abs(A[h] - Ah)) < 1e-10
        assert jnp.max(jnp.abs(B[h] - Bh)) < 1e-10
