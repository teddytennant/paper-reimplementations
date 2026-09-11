"""CPU checks of DKRGD identities from arXiv:2609.11712."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkrgd.distributed import fit_dkrgd, partition, predict_dkrgd
from dkrgd.kernel import gaussian_kernel, kernel_matrix
from dkrgd.krgd import fit_krgd, predict_krgd, step_size_bound
from dkrgd.losses import (
    C_G,
    G,
    G_PRIME_PLUS_0,
    G_prime,
    G_prime_plus_0,
    influence_weight,
    l_sigma,
)

LOSSES = ("huber", "fair", "cauchy", "welsch")


def test_l_sigma_zero_at_origin():
    # l_σ(u) = G(u²/σ²) and G(0) = 0 in Examples 1–4.
    for loss in LOSSES:
        val = float(l_sigma(0.0, sigma=1.7, loss=loss))
        assert val == pytest.approx(0.0, abs=1e-7)
        assert float(G(0.0, loss=loss)) == pytest.approx(0.0, abs=1e-7)


def test_loss_derivative_at_zero_and_g_prime_plus_0():
    # l_σ is even-powered in u through G(u²/σ²), so l'_σ(0) = 0.
    # Condition (1): G'_+(0) > 0; for Examples 1–4 this is 1/2.
    sigma = 1.3
    eps = 1e-4
    for loss in LOSSES:
        assert G_prime_plus_0(loss) == pytest.approx(0.5)
        gp0 = float(G_prime(0.0, loss=loss))
        assert gp0 == pytest.approx(G_PRIME_PLUS_0, abs=1e-6)
        plus = float(l_sigma(eps, sigma, loss=loss))
        minus = float(l_sigma(-eps, sigma, loss=loss))
        deriv = (plus - minus) / (2.0 * eps)
        assert deriv == pytest.approx(0.0, abs=1e-6)


def test_g_prime_positive_and_cg_bound():
    # (1): G'(s) > 0 for s > 0 and C_G = sup |G'| < ∞ (here 1/2).
    s = jnp.linspace(1e-4, 20.0, 50)
    for loss in LOSSES:
        gp = np.asarray(G_prime(s, loss=loss))
        assert np.all(gp > 0.0)
        assert float(np.max(gp)) <= C_G + 1e-6


def test_l_sigma_matches_example_closed_forms():
    # Examples 1–4: l_σ(u) equals the listed closed form, not only G(u²/σ²).
    u = jnp.array([-2.0, -0.4, 0.0, 0.4, 2.0])
    sigma = 0.7
    s = (u / sigma) ** 2
    np.testing.assert_allclose(
        np.asarray(l_sigma(u, sigma, "huber")),
        np.asarray(G(s, "huber")),
        atol=1e-6,
    )
    absu = np.abs(np.asarray(u))
    huber_closed = np.where(
        absu <= sigma,
        absu**2 / (2.0 * sigma**2),
        absu / sigma - 0.5,
    )
    np.testing.assert_allclose(np.asarray(l_sigma(u, sigma, "huber")), huber_closed, atol=1e-6)
    fair_closed = absu / sigma - np.log(1.0 + absu / sigma)
    np.testing.assert_allclose(np.asarray(l_sigma(u, sigma, "fair")), fair_closed, atol=1e-6)
    cauchy_closed = np.log(1.0 + np.asarray(u) ** 2 / (2.0 * sigma**2))
    np.testing.assert_allclose(np.asarray(l_sigma(u, sigma, "cauchy")), cauchy_closed, atol=1e-6)
    welsch_closed = 1.0 - np.exp(-np.asarray(u) ** 2 / (2.0 * sigma**2))
    np.testing.assert_allclose(np.asarray(l_sigma(u, sigma, "welsch")), welsch_closed, atol=1e-6)


def test_redescending_cauchy_welsch_and_huber_bounded():
    # Update weight in (4) is G'(ξ)(f−y). Cauchy/Welsch: → 0 as |u|→∞.
    # Huber: G'(ξ)u → (σ/2) sign(u), bounded but not redescending.
    sigma = 0.8
    large = jnp.array([0.0, 1.0, 10.0, 1e3, 1e5])
    for loss in ("cauchy", "welsch"):
        w = np.abs(np.asarray(influence_weight(large, sigma, loss=loss)))
        assert w[0] == pytest.approx(0.0, abs=1e-12)
        assert w[-1] < w[2]
        assert w[-1] < 1e-3
    huber = np.asarray(influence_weight(jnp.array([1e6, -1e6]), sigma, loss="huber"))
    np.testing.assert_allclose(huber, [0.5 * sigma, -0.5 * sigma], atol=1e-5)


def test_gaussian_kernel_unit_diagonal_and_spsd():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (7, 3))
    K = np.asarray(kernel_matrix(X, lengthscale=0.9))
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6)
    evals = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert evals.min() >= -1e-5
    x0 = X[0]
    assert float(gaussian_kernel(x0, x0, lengthscale=0.9)) == pytest.approx(1.0)


def test_one_local_gd_step_matches_hand_two_point_update():
    # One step of (4) on two points, Welsch G'(s) = (1/2) exp(−s/2).
    X = jnp.array([[0.0], [1.0]])
    y = jnp.array([0.0, 1.0])
    sigma = 1.0
    eta = 0.4
    lengthscale = 1.0
    model = fit_krgd(
        X,
        y,
        n_steps=1,
        eta=eta,
        sigma=sigma,
        loss="welsch",
        lengthscale=lengthscale,
    )
    n = 2
    resid0 = -y  # f_1 = 0
    xi = (resid0 / sigma) ** 2
    gp = 0.5 * jnp.exp(-0.5 * xi)
    alpha_hand = -(eta / n) * gp * resid0
    np.testing.assert_allclose(np.asarray(model.alpha), np.asarray(alpha_hand), atol=1e-6)

    # Predictions use the representer expansion with the Gaussian Gram matrix.
    K = kernel_matrix(X, lengthscale=lengthscale)
    pred_hand = K @ alpha_hand
    pred = predict_krgd(model, X)
    np.testing.assert_allclose(np.asarray(pred), np.asarray(pred_hand), atol=1e-6)


def test_m1_recovers_centralized_krgd():
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (12, 2))
    y = jnp.sin(X[:, 0]) + 0.1 * X[:, 1]
    kwargs = dict(n_steps=5, eta=0.3, sigma=1.2, loss="cauchy", lengthscale=0.8)
    central = fit_krgd(X, y, **kwargs)
    dist = fit_dkrgd(X, y, m=1, **kwargs)
    Xq = jax.random.normal(jax.random.PRNGKey(2), (6, 2))
    np.testing.assert_allclose(
        np.asarray(predict_krgd(central, Xq)),
        np.asarray(predict_dkrgd(dist, Xq)),
        atol=1e-5,
    )


def test_identical_local_solutions_average_to_themselves():
    # If every local estimator equals f, the weighted average (5) equals f.
    key = jax.random.PRNGKey(3)
    X = jax.random.uniform(key, (16, 1), minval=-1.0, maxval=1.0)
    y = 0.5 * X[:, 0]
    kwargs = dict(n_steps=4, eta=0.25, sigma=1.0, loss="huber", lengthscale=0.6)
    one = fit_krgd(X, y, **kwargs)
    copies = fit_dkrgd(
        jnp.concatenate([X, X, X], axis=0),
        jnp.concatenate([y, y, y], axis=0),
        m=3,
        **kwargs,
    )
    # Three identical copies of the same sample: each local fit equals `one`.
    Xq = jnp.linspace(-1.0, 1.0, 9)[:, None]
    for loc in copies:
        np.testing.assert_allclose(
            np.asarray(loc.model.alpha), np.asarray(one.alpha), atol=1e-5
        )
        assert loc.weight == pytest.approx(1.0 / 3.0)
    pred_avg = predict_dkrgd(copies, Xq)
    pred_one = predict_krgd(one, Xq)
    np.testing.assert_allclose(np.asarray(pred_avg), np.asarray(pred_one), atol=1e-5)


def test_partition_covers_and_is_disjoint():
    X = jnp.arange(10.0)[:, None]
    y = jnp.arange(10.0)
    shards = partition(X, y, m=3)
    sizes = [int(s[1].shape[0]) for s in shards]
    assert sizes == [4, 3, 3]
    stacked = np.concatenate([np.asarray(s[1]) for s in shards])
    np.testing.assert_array_equal(stacked, np.arange(10.0))


def test_finite_fit_on_noisy_sine():
    key = jax.random.PRNGKey(4)
    x = jnp.linspace(-jnp.pi, jnp.pi, 24)
    X = x[:, None]
    y = jnp.sin(x) + 0.2 * jax.random.normal(key, (24,))
    eta = 0.5 * step_size_bound(kappa=1.0)
    locals_ = fit_dkrgd(
        X, y, m=3, n_steps=6, eta=eta, sigma=1.0, loss="fair", lengthscale=0.7
    )
    pred = np.asarray(predict_dkrgd(locals_, X))
    assert np.all(np.isfinite(pred))
    assert pred.shape == (24,)
    # Better than predicting 0 on this tiny in-sample set.
    assert float(np.mean((pred - np.asarray(y)) ** 2)) < float(np.mean(np.asarray(y) ** 2))
