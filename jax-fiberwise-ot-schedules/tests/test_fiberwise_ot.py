"""Allocation identities, path lengths, Φ roundtrip, and 1D training smoke."""

import jax
import jax.numpy as jnp
import numpy as np

from fiberwise_ot.allocate import (
    LAMBDA_DDPM,
    LAMBDA_FM,
    allocate_weights,
    discrete_action,
    phi_from_weights,
    apply_phi,
    apply_phi_inv,
    solve_eta,
)
from fiberwise_ot.paths import (
    SOFF,
    alphabar,
    alpha_sigma,
    cosine_vp_arc_length,
    cond_ot_arc_length,
    cond_ot_gamma,
    phi_cosine,
)
from fiberwise_ot.train import one_shot_retrain, train_loop


def test_cosine_vp_formulas():
    t = jnp.array(0.3)
    phi = phi_cosine(t)
    expected_phi = (jnp.pi / 2.0) * (t + SOFF) / (1.0 + SOFF)
    np.testing.assert_allclose(phi, expected_phi, rtol=1e-6)
    ab = alphabar(t)
    expected_ab = jnp.cos(phi) ** 2 / jnp.cos(phi_cosine(0.0)) ** 2
    np.testing.assert_allclose(ab, expected_ab, rtol=1e-6)
    alpha, sigma = alpha_sigma(t)
    np.testing.assert_allclose(alpha, jnp.sqrt(ab), rtol=1e-6)
    np.testing.assert_allclose(sigma, jnp.sqrt(1.0 - ab), rtol=1e-6)


def test_cosine_vp_L_k_positive_and_sum():
    t_edges = jnp.linspace(0.0, 1.0, 64)
    L = cosine_vp_arc_length(t_edges)
    assert jnp.all(L > 0)
    ab0 = alphabar(t_edges[0])
    ab1 = alphabar(t_edges[-1])
    # ᾱ(0) = 1 so arccos(√ᾱ(0)) = 0; telescoping sum is arccos(√ᾱ(1)) − arccos(√ᾱ(0)).
    total = jnp.arccos(jnp.sqrt(ab1)) - jnp.arccos(jnp.sqrt(ab0))
    np.testing.assert_allclose(jnp.sum(L), total, rtol=1e-5)
    np.testing.assert_allclose(ab0, 1.0, atol=1e-6)
    np.testing.assert_allclose(jnp.arccos(jnp.sqrt(ab0)), 0.0, atol=1e-6)


def test_cond_ot_L_k_sum():
    tau_edges = jnp.linspace(0.0, 1.0, 51)
    L = cond_ot_arc_length(tau_edges)
    np.testing.assert_allclose(jnp.sum(L), jnp.sqrt(2.0), rtol=1e-6)
    m, s = cond_ot_gamma(tau_edges)
    np.testing.assert_allclose(m + s, jnp.ones_like(m), rtol=1e-6)
    np.testing.assert_allclose(L, jnp.sqrt(2.0) * jnp.diff(tau_edges), rtol=1e-6)


def test_allocate_sums_to_one_nonnegative():
    L = jnp.array([0.2, 0.3, 0.5])
    r = jnp.array([1.0, 4.0, 9.0])
    lam = 2.0
    w = allocate_weights(L, r, lam)
    np.testing.assert_allclose(jnp.sum(w), 1.0, rtol=1e-6)
    assert jnp.all(w >= 0)
    eta = solve_eta(L, r, lam)
    expected = L / jnp.sqrt(eta + lam * r)
    expected = expected / jnp.sum(expected)
    np.testing.assert_allclose(w, expected, rtol=1e-5)
    J = discrete_action(L, w, r, lam)
    expected_J = jnp.sum(L**2 / w + lam * w * r)
    np.testing.assert_allclose(J, expected_J, rtol=1e-6)
    assert jnp.isfinite(J)


def test_allocate_kinetic_baseline_when_r_zero_or_lam_zero():
    L = jnp.array([1.0, 3.0, 6.0])
    r = jnp.zeros(3)
    w = allocate_weights(L, r, 10.0)
    np.testing.assert_allclose(w, L / jnp.sum(L), rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(w), 1.0, rtol=1e-6)
    w_lam0 = allocate_weights(L, jnp.array([1.0, 4.0, 9.0]), 0.0)
    np.testing.assert_allclose(w_lam0, L / jnp.sum(L), rtol=1e-6)


def test_constant_r_gives_w_proportional_to_L():
    L = jnp.array([0.1, 0.4, 0.2, 0.3])
    r = jnp.full_like(L, 3.7)
    w = allocate_weights(L, r, LAMBDA_FM)
    np.testing.assert_allclose(w, L / jnp.sum(L), rtol=1e-5)


def test_constant_L_higher_r_gets_less_time():
    L = jnp.ones(5)
    r = jnp.array([0.2, 0.5, 1.0, 2.0, 4.0])
    lam = 3.0
    w = allocate_weights(L, r, lam)
    eta = float(solve_eta(L, r, lam))
    ratio = w[0] / w[-1]
    expected_ratio = np.sqrt((eta + lam * float(r[-1])) / (eta + lam * float(r[0])))
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-4)
    np.testing.assert_array_equal(np.argsort(np.asarray(w)), np.argsort(np.asarray(r))[::-1])


def test_w_not_uniform_when_r_not_constant():
    tau_edges = jnp.linspace(0.0, 1.0, 9)
    L = cond_ot_arc_length(tau_edges)
    r = jnp.linspace(0.1, 2.0, L.shape[0])
    w = allocate_weights(L, r, LAMBDA_FM)
    np.testing.assert_allclose(jnp.sum(w), 1.0, rtol=1e-6)
    assert not np.allclose(w, jnp.full_like(w, 1.0 / w.shape[0]), rtol=1e-3, atol=1e-3)
    w_const = allocate_weights(L, jnp.ones_like(r), LAMBDA_FM)
    np.testing.assert_allclose(w_const, jnp.full_like(w, 1.0 / w.shape[0]), rtol=1e-5)


def test_allocate_minimizes_discrete_action():
    L = jnp.array([0.2, 0.3, 0.5])
    r = jnp.array([1.0, 4.0, 9.0])
    lam = 2.0
    w = allocate_weights(L, r, lam)
    J_star = discrete_action(L, w, r, lam)
    w_uniform = jnp.full_like(w, 1.0 / w.shape[0])
    assert J_star < discrete_action(L, w_uniform, r, lam)
    key = jax.random.PRNGKey(0)
    for _ in range(8):
        key, k = jax.random.split(key)
        logits = jax.random.normal(k, w.shape)
        w_rand = jax.nn.softmax(logits)
        assert J_star < discrete_action(L, w_rand, r, lam)


def test_lambda_defaults():
    assert LAMBDA_DDPM == 220
    assert LAMBDA_FM == 450


def test_phi_roundtrip():
    L = jnp.array([0.1, 0.4, 0.2, 0.3])
    r = jnp.array([0.5, 1.5, 0.8, 2.0])
    w = allocate_weights(L, r, LAMBDA_DDPM)
    tau_edges, t_edges = phi_from_weights(w)
    np.testing.assert_allclose(tau_edges[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(tau_edges[-1], 1.0, atol=1e-6)
    np.testing.assert_allclose(t_edges[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(t_edges[-1], 1.0, atol=1e-6)
    np.testing.assert_allclose(apply_phi(tau_edges, tau_edges, t_edges), t_edges, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(apply_phi_inv(t_edges, tau_edges, t_edges), tau_edges, rtol=1e-6, atol=1e-6)
    tau = jnp.linspace(0.0, 1.0, 21)
    t = apply_phi(tau, tau_edges, t_edges)
    tau_back = apply_phi_inv(t, tau_edges, t_edges)
    np.testing.assert_allclose(tau_back, tau, rtol=1e-5, atol=1e-6)


def test_1d_smoke_finite_loss():
    key = jax.random.PRNGKey(0)
    _, loss_ddpm = train_loop(key, family="ddpm", steps=8, batch=32, lr=3e-3)
    _, loss_fm = train_loop(key, family="fm", steps=8, batch=32, lr=3e-3)
    assert jnp.isfinite(loss_ddpm)
    assert jnp.isfinite(loss_fm)


def test_one_shot_retrain_finite():
    key = jax.random.PRNGKey(1)
    out = one_shot_retrain(
        key,
        family="ddpm",
        baseline_steps=6,
        retrain_steps=6,
        batch=32,
        n_intervals=8,
        n_risk_samples=64,
    )
    assert jnp.isfinite(out["baseline_loss"])
    assert jnp.isfinite(out["retrain_loss"])
    np.testing.assert_allclose(jnp.sum(out["w"]), 1.0, rtol=1e-5)
    assert jnp.all(out["w"] >= 0)
    r = out["r"]
    w = out["w"]
    if not np.allclose(r, jnp.full_like(r, float(r[0])), rtol=1e-3, atol=1e-3):
        assert not np.allclose(w, jnp.full_like(w, 1.0 / w.shape[0]), rtol=1e-3, atol=1e-3)
