"""CPU float64 checks of the Kalman Delta Network recurrences."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from kalman_delta.delta import delta_rule_step, isotropic_gain_scale, kappa_from_beta
from kalman_delta.diagonal import (
    diagonal_gain,
    diagonal_step,
    mobius_scan,
    sequential_p,
    variational_diagonal_p,
)
from kalman_delta.isotropic import isotropic_filter, isotropic_predict_scale, isotropic_step
from kalman_delta.kalman import kalman_filter, kalman_step, read


def _scalar_kalman(s, p, d, omega, k, v, r):
    s_hat = d * s
    p_hat = d * p * d + omega
    kappa = p_hat * k / (r + k * p_hat * k)
    s_new = s_hat + kappa * (v - s_hat * k)
    p_new = (1.0 - kappa * k) * p_hat
    return s_new, p_new, kappa


def test_exact_kalman_matches_scalar_filter():
    rng = np.random.default_rng(0)
    T = 16
    s = 0.0
    p = 1.0
    S = jnp.array([[s]], dtype=jnp.float64)
    P = jnp.array([[p]], dtype=jnp.float64)
    Ds, Omegas, ks, vs, rs = [], [], [], [], []
    s_seq, p_seq, k_seq = [], [], []
    for _ in range(T):
        d = float(rng.uniform(0.4, 0.99))
        omega = float(rng.uniform(0.01, 0.2))
        k = float(rng.normal())
        v = float(rng.normal())
        r = float(rng.uniform(0.05, 0.5))
        s, p, kappa = _scalar_kalman(s, p, d, omega, k, v, r)
        s_seq.append(s)
        p_seq.append(p)
        k_seq.append(kappa)
        Ds.append([[d]])
        Omegas.append([[omega]])
        ks.append([k])
        vs.append([v])
        rs.append(r)
    S_out, P_out, kappa_out = kalman_filter(
        jnp.array([[0.0]], dtype=jnp.float64),
        jnp.array([[1.0]], dtype=jnp.float64),
        jnp.array(Ds, dtype=jnp.float64),
        jnp.array(Omegas, dtype=jnp.float64),
        jnp.array(ks, dtype=jnp.float64),
        jnp.array(vs, dtype=jnp.float64),
        jnp.array(rs, dtype=jnp.float64),
    )
    np.testing.assert_allclose(np.array(s_seq), np.array(S_out[:, 0, 0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(p_seq), np.array(P_out[:, 0, 0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(k_seq), np.array(kappa_out[:, 0]), rtol=1e-12, atol=1e-12)


def test_exact_kalman_matches_batch_closed_form_1d():
    # One-step closed form for d_k = d_v = 1: same as the scalar Kalman update.
    S = jnp.array([[0.3]], dtype=jnp.float64)
    P = jnp.array([[1.4]], dtype=jnp.float64)
    D = jnp.array([[0.8]], dtype=jnp.float64)
    Omega = jnp.array([[0.05]], dtype=jnp.float64)
    k = jnp.array([1.2], dtype=jnp.float64)
    v = jnp.array([-0.4], dtype=jnp.float64)
    r = jnp.array(0.2, dtype=jnp.float64)
    S_new, P_new, kappa = kalman_step(S, P, D, Omega, k, v, r)
    s, p, kap = _scalar_kalman(0.3, 1.4, 0.8, 0.05, 1.2, -0.4, 0.2)
    np.testing.assert_allclose(float(S_new[0, 0]), s, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(float(P_new[0, 0]), p, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(float(kappa[0]), kap, rtol=1e-14, atol=1e-14)


def test_delta_rule_equals_isotropic_kalman_mean():
    rng = np.random.default_rng(1)
    d_k, d_v = 5, 3
    S = jnp.array(rng.normal(size=(d_k, d_v)), dtype=jnp.float64)
    k = jnp.array(rng.normal(size=(d_k,)), dtype=jnp.float64)
    v = jnp.array(rng.normal(size=(d_v,)), dtype=jnp.float64)
    r = 0.3
    b_hat = 0.7
    D = jnp.eye(d_k, dtype=jnp.float64)
    P_hat = b_hat * jnp.eye(d_k, dtype=jnp.float64)
    S_hat = D @ S
    from kalman_delta.kalman import kalman_update

    S_kalman, _, kappa = kalman_update(S_hat, P_hat, k, v, r)
    beta = isotropic_gain_scale(b_hat, k, r)
    kappa_delta = kappa_from_beta(beta, k)
    S_delta = delta_rule_step(S, k, v, beta, D=D)
    np.testing.assert_allclose(np.array(kappa), np.array(kappa_delta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(S_kalman), np.array(S_delta), rtol=1e-12, atol=1e-12)


def test_diagonal_variational_p_and_mean_match_dense_one_step():
    rng = np.random.default_rng(2)
    d_k, d_v = 6, 4
    p = jnp.array(rng.uniform(0.2, 1.5, size=(d_k,)), dtype=jnp.float64)
    alpha = jnp.array(rng.uniform(0.3, 0.95, size=(d_k,)), dtype=jnp.float64)
    omega = jnp.array(rng.uniform(0.01, 0.2, size=(d_k,)), dtype=jnp.float64)
    k = jnp.array(rng.normal(size=(d_k,)), dtype=jnp.float64)
    v = jnp.array(rng.normal(size=(d_v,)), dtype=jnp.float64)
    r = 0.25
    S = jnp.array(rng.normal(size=(d_k, d_v)), dtype=jnp.float64)

    S_diag, p_new, kappa_diag, p_hat = diagonal_step(S, p, alpha, omega, k, v, r, mu=1.0)

    p_closed = 1.0 / (1.0 / p_hat + (k * k) / r)
    np.testing.assert_allclose(np.array(p_new), np.array(p_closed), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.array(p_new),
        np.array(variational_diagonal_p(p_hat, k, r, mu=1.0)),
        rtol=1e-12,
        atol=1e-12,
    )

    D = jnp.diag(alpha)
    Omega = jnp.diag(omega)
    P = jnp.diag(p)
    S_exact, P_exact, kappa_exact = kalman_step(S, P, D, Omega, k, v, r)
    np.testing.assert_allclose(np.array(S_diag), np.array(S_exact), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(kappa_diag), np.array(kappa_exact), rtol=1e-12, atol=1e-12)
    # Projection keeps only the diagonal of the dense posterior precision inverse.
    assert P_exact.shape == (d_k, d_k)
    off = np.array(P_exact) - np.diag(np.diag(np.array(P_exact)))
    assert np.linalg.norm(off) > 1e-8


def test_mobius_scan_matches_sequential():
    rng = np.random.default_rng(3)
    T, d_k = 32, 7
    alpha = jnp.array(rng.uniform(0.4, 0.99, size=(T, d_k)), dtype=jnp.float64)
    omega = jnp.array(rng.uniform(0.01, 0.3, size=(T, d_k)), dtype=jnp.float64)
    k = jnp.array(rng.normal(size=(T, d_k)), dtype=jnp.float64)
    r = jnp.array(rng.uniform(0.1, 0.8, size=(T, 1)), dtype=jnp.float64)
    mu = 1.0
    u = (mu / r) * (k * k)
    p_scan = mobius_scan(alpha, omega, u)
    p_seq = sequential_p(alpha, omega, u)
    np.testing.assert_allclose(np.array(p_scan), np.array(p_seq), rtol=1e-11, atol=1e-11)


def test_information_scale_mu():
    rng = np.random.default_rng(4)
    d_k, d_v = 5, 2
    p = jnp.ones((d_k,), dtype=jnp.float64)
    alpha = jnp.array(rng.uniform(0.5, 0.9, size=(d_k,)), dtype=jnp.float64)
    omega = jnp.array(rng.uniform(0.05, 0.2, size=(d_k,)), dtype=jnp.float64)
    k = jnp.array(rng.normal(size=(d_k,)), dtype=jnp.float64)
    v = jnp.array(rng.normal(size=(d_v,)), dtype=jnp.float64)
    r = 0.4
    S = jnp.array(rng.normal(size=(d_k, d_v)), dtype=jnp.float64)

    S1, p1, kappa1, p_hat1 = diagonal_step(S, p, alpha, omega, k, v, r, mu=1.0)
    S2, p2, kappa2, p_hat2 = diagonal_step(S, p, alpha, omega, k, v, r, mu=4.0)

    np.testing.assert_allclose(np.array(kappa1), np.array(kappa2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(S1), np.array(S2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(p_hat1), np.array(p_hat2), rtol=1e-12, atol=1e-12)
    assert np.all(np.array(p2) < np.array(p1) - 1e-12)

    p_var = variational_diagonal_p(p_hat1, k, r, mu=1.0)
    np.testing.assert_allclose(np.array(p1), np.array(p_var), rtol=1e-12, atol=1e-12)


def test_isotropic_beta_and_memory_match_loop():
    rng = np.random.default_rng(5)
    T, d_k, d_v = 12, 4, 3
    S0 = jnp.array(rng.normal(size=(d_k, d_v)), dtype=jnp.float64)
    b0 = 1.0
    alpha = jnp.array(rng.uniform(0.4, 0.95, size=(T, d_k)), dtype=jnp.float64)
    omega = jnp.array(rng.uniform(0.02, 0.15, size=(T,)), dtype=jnp.float64)
    k = jnp.array(rng.normal(size=(T, d_k)), dtype=jnp.float64)
    v = jnp.array(rng.normal(size=(T, d_v)), dtype=jnp.float64)
    r = jnp.array(rng.uniform(0.1, 0.6, size=(T,)), dtype=jnp.float64)

    S_scan, b_scan, beta_scan = isotropic_filter(S0, b0, alpha, omega, k, v, r)

    S = np.array(S0)
    b = b0
    S_loop, b_loop, beta_loop = [], [], []
    for t in range(T):
        a_t = np.array(alpha[t])
        k_t = np.array(k[t])
        v_t = np.array(v[t])
        a = np.mean(a_t * a_t)
        b_hat = a * b + float(omega[t])
        k2 = float(np.dot(k_t, k_t))
        beta = b_hat / (float(r[t]) + b_hat * k2)
        S_hat = a_t[:, None] * S
        innov = v_t - S_hat.T @ k_t
        S = S_hat + beta * np.outer(k_t, innov)
        b = b_hat / (1.0 + b_hat * k2 / (d_k * float(r[t])))
        S_loop.append(S)
        b_loop.append(b)
        beta_loop.append(beta)

    np.testing.assert_allclose(np.array(S_scan), np.stack(S_loop), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(b_scan), np.array(b_loop), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.array(beta_scan), np.array(beta_loop), rtol=1e-12, atol=1e-12)

    # Single-step β formula vs eq 40.
    a, b_hat = isotropic_predict_scale(1.0, alpha[0], omega[0])
    beta = isotropic_step(S0, 1.0, alpha[0], omega[0], k[0], v[0], r[0])[2]
    k2 = jnp.dot(k[0], k[0])
    np.testing.assert_allclose(
        float(beta),
        float(b_hat / (r[0] + b_hat * k2)),
        rtol=1e-14,
        atol=1e-14,
    )


def test_read_is_S_transpose_q():
    rng = np.random.default_rng(6)
    d_k, d_v = 8, 5
    S = jnp.array(rng.normal(size=(d_k, d_v)), dtype=jnp.float64)
    q = jnp.array(rng.normal(size=(d_k,)), dtype=jnp.float64)
    o = read(S, q)
    expected = S.T @ q
    np.testing.assert_allclose(np.array(o), np.array(expected), rtol=1e-14, atol=1e-14)
    assert o.shape == (d_v,)
