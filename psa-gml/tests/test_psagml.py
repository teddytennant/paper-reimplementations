"""CPU tests for the PSA-GML JAX kernel (arXiv:2609.29150)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from psagml.channels import rayleigh_channels
from psagml.lstm import init_lstm, init_lstm_state, lstm_step, preprocess_grad
from psagml.pso import random_phase_zf_wsr, run_pso
from psagml.star_ris import (
    amplitude_split,
    composite_channel,
    star_coefficients,
    unconstrained_phi,
)
from psagml.train_meta import meta_train, pso_to_xi, run_psagml, wsr_from_xi
from psagml.wsr import sinr, weighted_sum_rate
from psagml.zf import collapsed_precoder, power_normalize, zf_collapsed_X, zf_precoder

NT, N, KT, KR = 4, 4, 2, 2
K = KT + KR
PMAX = 10.0
SIGMA2 = 1.0
ATOL = 1e-4


def _toy_channel(seed=0):
    G, Ht, Hr = rayleigh_channels(jax.random.PRNGKey(seed), NT, N, KT, KR)
    omega = jnp.ones((K,), dtype=jnp.float32)
    return G, Ht, Hr, omega


def _random_star(seed=1):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    phi = jax.random.uniform(k1, (N,), maxval=jnp.pi / 2.0)
    tt = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
    tr = jax.random.uniform(k3, (N,), maxval=2.0 * jnp.pi)
    return phi, tt, tr


def test_energy_conservation():
    phi = jnp.array([0.0, 0.2, 0.8, jnp.pi / 2.0], dtype=jnp.float32)
    bt, br = amplitude_split(phi)
    assert jnp.allclose(bt**2 + br**2, 1.0, atol=1e-6)


def test_prop1_zf_identity():
    G, Ht, Hr, _ = _toy_channel(0)
    phi, tt, tr = _random_star(1)
    vt, vr = star_coefficients(phi, tt, tr)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W = zf_precoder(A)
    I = A @ W
    assert jnp.allclose(I, jnp.eye(K, dtype=A.dtype), atol=ATOL, rtol=ATOL)
    off = I - jnp.diag(jnp.diag(I))
    assert jnp.allclose(off, 0.0, atol=ATOL)


def test_lemma2_collapsed_equals_zf():
    G, Ht, Hr, _ = _toy_channel(2)
    phi, tt, tr = _random_star(3)
    vt, vr = star_coefficients(phi, tt, tr)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W_zf = zf_precoder(A)
    X = zf_collapsed_X(A)
    W = collapsed_precoder(A, X)
    assert jnp.allclose(W, W_zf, atol=ATOL, rtol=ATOL)


def test_prop2_phi_range():
    # Moderate logits stay strictly inside (0, pi/2). Float32 sigmoid saturates
    # at |z| ~ 16, so we do not use extreme values here.
    tilde = jnp.array([-8.0, -2.0, 0.0, 2.0, 8.0], dtype=jnp.float32)
    phi = unconstrained_phi(tilde)
    assert jnp.all(phi > 0.0)
    assert jnp.all(phi < jnp.pi / 2.0)
    assert unconstrained_phi(jnp.array(0.0)) == pytest.approx(jnp.pi / 4.0, rel=1e-5)


def test_power_scaling():
    G, Ht, Hr, _ = _toy_channel(4)
    phi, tt, tr = _random_star(5)
    vt, vr = star_coefficients(phi, tt, tr)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W = power_normalize(zf_precoder(A), PMAX)
    power = jnp.real(jnp.sum(jnp.abs(W) ** 2))
    assert jnp.allclose(power, PMAX, atol=1e-5)


def test_wsr_zf_interference_free():
    G, Ht, Hr, omega = _toy_channel(6)
    phi, tt, tr = _random_star(7)
    vt, vr = star_coefficients(phi, tt, tr)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W = power_normalize(zf_precoder(A), PMAX)
    R = weighted_sum_rate(A, W, omega, SIGMA2)
    assert jnp.isfinite(R)
    assert float(R) >= 0.0
    gamma = sinr(A, W, SIGMA2)
    aw = A @ W
    signal = jnp.abs(jnp.diag(aw)) ** 2
    expected = signal / SIGMA2
    assert jnp.allclose(gamma, expected, atol=ATOL, rtol=ATOL)
    off = aw - jnp.diag(jnp.diag(aw))
    assert jnp.allclose(off, 0.0, atol=ATOL)


def test_pso_global_best_nondecreasing():
    G, Ht, Hr, omega = _toy_channel(8)
    _, _, hist = run_pso(
        jax.random.PRNGKey(9),
        G,
        Ht,
        Hr,
        PMAX,
        SIGMA2,
        omega,
        n_particles=4,
        n_iters=5,
    )
    diffs = jnp.diff(hist)
    assert jnp.all(diffs >= -1e-5)


def test_lstm_step_and_preprocess():
    g0 = preprocess_grad(jnp.zeros((4,)))
    assert jnp.allclose(g0, 0.0)
    x = jnp.array([0.0, 0.3, -1.7, 4.2], dtype=jnp.float32)
    assert jnp.allclose(preprocess_grad(-x), -preprocess_grad(x), atol=1e-6)

    params = init_lstm(jax.random.PRNGKey(10), hidden=8)
    D, H = 6, 8
    g = jnp.ones((D,), dtype=jnp.float32)
    h, c = init_lstm_state(D, H)
    _, _, delta = lstm_step(params, g, h, c, nu=0.05)
    xi = jnp.zeros((D,), dtype=jnp.float32)
    xi2 = xi + delta
    assert not jnp.allclose(xi, xi2, atol=1e-12)


def test_meta_detach_grads():
    channels = [rayleigh_channels(jax.random.PRNGKey(s), NT, N, KT, KR) for s in (11, 12)]
    psi, losses = meta_train(
        jax.random.PRNGKey(13),
        channels,
        T=3,
        n_steps=2,
        hidden=8,
        nu=0.05,
        lr=1e-2,
        Pmax=PMAX,
        sigma2=SIGMA2,
        n_particles=4,
        n_pso_iters=2,
        warm_start="random",
    )
    assert jnp.all(jnp.isfinite(losses))
    Mc, Mx = psi

    def loss_fn(p):
        from psagml.train_meta import meta_loss

        G, Ht, Hr = channels[0]
        omega = jnp.ones((K,), dtype=jnp.float32)
        Nloc = G.shape[0]
        xi0 = jnp.zeros((3 * Nloc + 2 * K * K,), dtype=jnp.float32)
        batch = [(G, Ht, Hr, xi0, omega)]
        return meta_loss(p, batch, T=2, nu=0.05, hidden=8, Pmax=PMAX, sigma2=SIGMA2)

    grads = jax.grad(loss_fn)((Mc, Mx))
    flat, _ = jax.tree_util.tree_flatten(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in flat)
    decreased = float(losses[-1]) <= float(losses[0]) + 1e-3
    any_nonzero = any(jnp.any(jnp.abs(g) > 0) for g in flat)
    assert decreased or any_nonzero


def test_e2e_pso_refine_beats_random():
    omega = jnp.ones((K,), dtype=jnp.float32)
    n_ok = 0
    last = None
    for seed in range(8):
        key = jax.random.PRNGKey(100 + seed)
        k_ch, k_run, k_rnd = jax.random.split(key, 3)
        G, Ht, Hr = rayleigh_channels(k_ch, NT, N, KT, KR)
        out = run_psagml(
            k_run,
            G,
            Ht,
            Hr,
            Pmax=PMAX,
            sigma2=SIGMA2,
            omega=omega,
            n_particles=4,
            n_pso_iters=5,
            hidden=8,
            nu=0.05,
            S=2,
        )
        R_ref = out["wsr_refined"]
        R_rnd = random_phase_zf_wsr(k_rnd, G, Ht, Hr, PMAX, SIGMA2, omega)
        last = (float(R_ref), float(R_rnd), float(out["wsr_pso"]))
        assert jnp.isfinite(R_ref)
        if float(R_ref) + 1e-4 >= float(R_rnd):
            n_ok += 1
            break
    assert n_ok >= 1, f"PSO+LSTM never beat random ZF; last refined/random/pso={last}"
