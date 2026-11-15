"""Checks for the lazy second-order steps. Nested loops use N and S overrides."""

import math

import jax.numpy as jnp
import numpy as np

from lazy_so import (
    CountingOracle,
    cubic_gradient,
    cubic_hessian,
    cubic_proximal,
    cubic_regularized_newton,
    extrapolation_params,
    finite_seg_search,
    lazy_iteration_cost,
    pagd_iteration_count,
    search_free_ms_acceleration,
)


def _cubic_value(d, gamma):
    d = np.asarray(d, dtype=np.float64)
    return (gamma / 3.0) * np.linalg.norm(d) ** 3


def test_cubic_matches_finite_differences():
    gamma = 1.3
    rng = np.random.default_rng(0)
    points = [rng.normal(size=3), np.zeros(3)]
    eps = 1e-6
    eye = np.eye(3)

    for d in points:
        analytic_g = np.asarray(cubic_gradient(d, gamma))
        analytic_h = np.asarray(cubic_hessian(d, gamma))
        fd_g = np.zeros(3)
        for i in range(3):
            fd_g[i] = (
                _cubic_value(d + eps * eye[i], gamma) - _cubic_value(d - eps * eye[i], gamma)
            ) / (2.0 * eps)
        fd_h = np.zeros((3, 3))
        for i in range(3):
            gp = np.asarray(cubic_gradient(d + eps * eye[i], gamma))
            gm = np.asarray(cubic_gradient(d - eps * eye[i], gamma))
            fd_h[:, i] = (gp - gm) / (2.0 * eps)
        if np.linalg.norm(d) == 0.0:
            assert np.all(analytic_g == 0.0)
            assert np.all(analytic_h == 0.0)
        assert np.allclose(analytic_g, fd_g, atol=1e-7, rtol=1e-6)
        assert np.allclose(analytic_h, fd_h, atol=1e-5, rtol=1e-5)


def _secular_newton(g, Hmat, H_reg, n_iter=40):
    """Independent 1D Newton solve of lambda = H_reg * ||(Hmat + lambda I)^{-1} g||."""
    Hmat = 0.5 * (Hmat + Hmat.T)
    evals, evecs = np.linalg.eigh(Hmat)
    g_hat = evecs.T @ g
    gnorm = np.linalg.norm(g_hat)
    if gnorm == 0.0 or H_reg == 0.0:
        s_hat = np.zeros_like(g_hat)
        mask = np.abs(evals) > 1e-12
        s_hat[mask] = -g_hat[mask] / evals[mask]
        return evecs @ s_hat
    mu_min = float(evals[0])
    lam = 0.5 * (-min(mu_min, 0.0) + math.sqrt(min(mu_min, 0.0) ** 2 + 4.0 * H_reg * gnorm))
    lam = max(lam, max(0.0, -mu_min) + 1e-8)
    for _ in range(n_iter):
        denom = evals + lam
        inv = 1.0 / denom
        sigma = float(np.linalg.norm(g_hat * inv))
        sigma_prime = -float(np.sum((g_hat**2) * (inv**3))) / sigma
        phi = sigma - lam / H_reg
        phi_prime = sigma_prime - 1.0 / H_reg
        step = phi / phi_prime
        lam = max(lam - step, max(0.0, -mu_min) + 1e-14)
        if abs(step) < 1e-14 * (1.0 + lam):
            break
    s = evecs @ (-g_hat / (evals + lam))
    return s


def test_crn_matches_secular_equation():
    # Closed form: (2 + lambda) s = -2, lambda = 2 |s|, s = (1 - sqrt(5)) / 2.
    closed = cubic_regularized_newton(
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        gamma=1.0,
        L=0.0,
        grad_f=lambda y: 2.0 * y + 2.0,
        hess_f=lambda y: jnp.asarray([[2.0]]),
    )
    s_closed = (1.0 - math.sqrt(5.0)) / 2.0
    assert abs(float(closed[0]) - s_closed) < 1e-8

    A = np.array(
        [[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 3.0]],
        dtype=np.float64,
    )
    b = np.array([0.4, -0.7, 0.2], dtype=np.float64)
    z = np.array([0.3, -0.5, 0.8], dtype=np.float64)
    x = np.zeros(3)
    gamma = 1.0
    L = 0.0
    A_j = jnp.asarray(A)
    b_j = jnp.asarray(b)

    def grad_f(y):
        return A_j @ y + b_j

    def hess_f(y):
        return A_j

    out = np.asarray(
        cubic_regularized_newton(jnp.asarray(z), jnp.asarray(x), gamma, L, grad_f, hess_f)
    )
    d = z - x
    radius = np.linalg.norm(d)
    g = A @ z + b + gamma * radius * d
    Hmat = A + gamma * (radius * np.eye(3) + np.outer(d, d) / radius)
    H_reg = L + 2.0 * gamma
    s_ind = _secular_newton(g, Hmat, H_reg)
    assert np.allclose(out, z + s_ind, atol=1e-5, rtol=1e-5)

    s = out - z
    lam = H_reg * np.linalg.norm(s)
    residual = Hmat @ s + g + lam * s
    assert np.linalg.norm(residual) < 1e-5


def test_finite_seg_search_quadratic_restriction():
    # phi(beta) = 0.5 * (beta - 0.3)^2, so phi'(beta) = beta - 0.3.
    zeta = 1e-4
    x = jnp.asarray([0.0])
    v = jnp.asarray([1.0])

    def grad_g(point):
        return point - 0.3

    y = finite_seg_search(grad_g, x, v, zeta)
    beta = float(y[0])
    slope = beta - 0.3
    assert 0.0 <= beta <= 1.0
    assert abs(slope) <= 1e-4
    assert -zeta - 1e-12 <= slope <= 1e-12

    # Endpoint: derivative already nonnegative at the left end.
    left = finite_seg_search(lambda p: p + 1.0, x, v, zeta)
    assert float(left[0]) == 0.0


def test_bisection_cap_returns():
    y = finite_seg_search(
        lambda p: p - 0.3,
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        zeta=0.0,
        max_steps=64,
    )
    assert 0.0 <= float(y[0]) <= 1.0


def test_cubic_proximal_moves_closer():
    x_star = jnp.asarray([0.2, -0.3, 0.4])
    x0 = jnp.asarray([1.5, -1.0, 0.7])

    def grad_f(y):
        return y - x_star

    def hess_f(y):
        return jnp.eye(3, dtype=jnp.float64)

    y = cubic_proximal(grad_f, hess_f, x0, gamma=1.0, L=0.0, N=4, S=1)
    start_dist = float(jnp.linalg.norm(x0 - x_star))
    end_dist = float(jnp.linalg.norm(y - x_star))
    start_f = 0.5 * start_dist**2
    end_f = 0.5 * end_dist**2
    assert end_dist < start_dist
    assert end_f < start_f


def test_query_counters_and_hessian_point():
    def f(x):
        return 0.5 * jnp.dot(x - 0.3, x - 0.3)

    oracle = CountingOracle(f)
    hess_points = []

    def hess(y):
        hess_points.append(np.asarray(y).copy())
        return oracle.hess(y)

    y = finite_seg_search(oracle.grad, jnp.asarray([0.0]), jnp.asarray([1.0]), 1e-4)
    assert oracle.n_grad > 0
    assert oracle.n_hess == 0
    assert 0.0 <= float(y[0]) <= 1.0

    z = jnp.asarray([0.4, -0.2])

    def grad_f(point):
        return oracle.grad(point)

    def hess_f(point):
        return hess(point)

    before = oracle.n_hess
    out = cubic_regularized_newton(z, jnp.zeros(2), 1.0, 0.0, grad_f, hess_f)
    assert oracle.n_hess == before + 1
    assert len(hess_points) == 1
    assert np.allclose(hess_points[0], np.asarray(z))
    assert out.shape == (2,)

    # A second call at a different point must read that point's Hessian.
    z2 = jnp.asarray([-0.7, 0.5])
    cubic_regularized_newton(z2, jnp.zeros(2), 1.0, 0.0, grad_f, hess_f)
    assert np.allclose(hess_points[-1], np.asarray(z2))
    assert not np.allclose(hess_points[0], hess_points[-1])


def test_lazy_iteration_cost():
    assert lazy_iteration_cost(10, 2, 5) == 20


def test_pagd_iteration_count_formula():
    # kappa = 64**3 makes log(kappa) / (3 log 64) = 1, so J = 1.
    # N = ceil(64 * 2^(2/3) * 64) = ceil(4096 * 2^(2/3)) = 6502.
    kappa = float(64**3)
    lam, J, c = extrapolation_params(kappa)
    assert lam == 1.0 / 64.0
    assert J == 1
    assert c == 2.0
    assert pagd_iteration_count(kappa, 1.0) == 6502
    assert pagd_iteration_count(2.0, 1.0) == 128
    # kappa = 1 still uses J = 1, N = ceil(64 * 2^(2/3)).
    assert pagd_iteration_count(1.0, 1.0) == math.ceil(64.0 * (2.0 ** (2.0 / 3.0)))


def test_algorithm1_t1_override_finishes():
    x_star = jnp.asarray([0.2, -0.1, 0.4, 0.3])
    x0 = jnp.zeros(4)

    def f(x):
        gap = x - x_star
        return 0.5 * jnp.dot(gap, gap)

    oracle = CountingOracle(f)
    y = search_free_ms_acceleration(
        oracle.grad, oracle.hess, x0, gamma=1.0, T=1, L=0.0, N=4, S=1
    )
    assert y.shape == (4,)
    assert math.isfinite(float(oracle.value(y)))
    assert float(oracle.value(y)) < float(f(x0))
    assert oracle.n_grad > 0
    assert oracle.n_hess > 0


def test_formula_n_proximal_t0():
    # One cubic proximal with the default N and S. kappa = 2 gives N = 128, S = 1.
    x_star = jnp.asarray([0.2, -0.1, 0.05])
    x0 = jnp.zeros(3)

    def grad_f(x):
        return x - x_star

    def hess_f(x):
        return jnp.eye(3, dtype=jnp.float64)

    y = search_free_ms_acceleration(grad_f, hess_f, x0, gamma=1.0, T=0, L=0.0)
    start = 0.5 * float(jnp.dot(x0 - x_star, x0 - x_star))
    end = 0.5 * float(jnp.dot(y - x_star, y - x_star))
    assert y.shape == (3,)
    assert end < start
