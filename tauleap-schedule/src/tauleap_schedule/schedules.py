"""Uniform, geometric, finite-K shooting, and limiting smooth schedules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tauleap_schedule.quad import as_numpy_fn, integrate


def uniform(K: int) -> np.ndarray:
    """β_k = k/K for k = 0, …, K."""
    if K < 1:
        raise ValueError("K must be >= 1")
    return np.linspace(0.0, 1.0, K + 1)


def geometric(K: int, a: float, N: int) -> np.ndarray:
    """Prop. 3: β_0 = 0, β_k = min{1, (1+a)^{k-1} a / N}, last knot forced to 1."""
    if K < 1:
        raise ValueError("K must be >= 1")
    k = np.arange(K + 1, dtype=np.float64)
    beta = np.minimum(1.0, np.power(1.0 + a, k - 1.0) * (a / float(N)))
    beta[0] = 0.0
    beta[-1] = 1.0
    return np.maximum.accumulate(np.clip(beta, 0.0, 1.0))


def reverse_geometric(K: int, a: float, N: int) -> np.ndarray:
    """Time-reversal of the geometric knot vector (still 0 → 1, increasing)."""
    g = geometric(K, a, N)
    return 1.0 - g[::-1]


def q_from_beta(beta) -> np.ndarray:
    """q_k = (β_k - β_{k-1}) / (1 - β_{k-1}), with q = 1 when the denom vanishes."""
    beta = np.asarray(beta, dtype=np.float64)
    d = np.diff(beta)
    denom = 1.0 - beta[:-1]
    q = np.divide(d, denom, out=np.ones_like(d), where=denom > 0.0)
    return q


def _eval_rho(rho_fn: Callable, x: float) -> float:
    rho = as_numpy_fn(rho_fn)
    return float(np.asarray(rho(np.array([x]))).reshape(-1)[0])


def _psi(b: float, delta: float, rho_fn: Callable, n_quad: int) -> float:
    """Ψ(b, δ) = (1/ρ(b)) ∫_{b-δ}^{b} ρ(u) du."""
    rb = _eval_rho(rho_fn, b)
    integ = integrate(rho_fn, b - delta, b, n=n_quad)
    if rb <= 0.0:
        return delta
    return integ / rb


def _forward(beta1: float, K: int, rho_fn: Callable, n_quad: int) -> np.ndarray:
    beta = np.zeros(K + 1, dtype=np.float64)
    beta[1] = beta1
    for k in range(1, K):
        delta = beta[k] - beta[k - 1]
        psi = _psi(float(beta[k]), float(delta), rho_fn, n_quad)
        beta[k + 1] = beta[k] + psi
        if not np.isfinite(beta[k + 1]) or beta[k + 1] > 2.0:
            beta[k + 1 :] = 2.0
            break
    return beta


def shoot_finite_k(
    rho_fn: Callable,
    K: int,
    n_quad: int = 1024,
    n_bisect: int = 80,
) -> np.ndarray:
    """Corollary 1 shooting: find β_1 so that β_K = 1 under the Ψ recurrence."""
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return np.array([0.0, 1.0])

    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        end = float(_forward(mid, K, rho_fn, n_quad)[-1])
        if end > 1.0:
            hi = mid
        else:
            lo = mid
    beta = _forward(0.5 * (lo + hi), K, rho_fn, n_quad)
    beta[0] = 0.0
    beta[-1] = 1.0
    return np.maximum.accumulate(np.clip(beta, 0.0, 1.0))


def stationary_k2(rho_fn: Callable, n_quad: int = 2048, n_bisect: int = 80) -> float:
    """K=2 stationarity: solve ∫_0^x ρ(u) du = (1-x) ρ(x)."""

    def f(x: float) -> float:
        integ = integrate(rho_fn, 0.0, x, n=n_quad)
        return integ - (1.0 - x) * _eval_rho(rho_fn, x)

    lo, hi = 1e-10, 1.0 - 1e-10
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _sqrt_g_prefix(g_fn: Callable, n_grid: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(0.0, 1.0, n_grid)
    g = as_numpy_fn(g_fn)(ys)
    gy = np.sqrt(np.maximum(g, 0.0))
    G = np.zeros_like(ys)
    G[1:] = np.cumsum(0.5 * (gy[1:] + gy[:-1]) * np.diff(ys))
    return ys, G


def G_sqrt(g_fn: Callable, y, n_grid: int = 4096) -> np.ndarray:
    """G(y) = ∫_0^y √g(s) ds, vectorized in y ∈ [0, 1]."""
    ys, G = _sqrt_g_prefix(g_fn, n_grid=n_grid)
    y = np.asarray(y, dtype=np.float64)
    return np.interp(np.clip(y, 0.0, 1.0), ys, G)


def limit_smooth(g_fn: Callable, t, n_grid: int = 4096) -> np.ndarray:
    """Corollary 2: β*(t) = G^{-1}(t G(1)) with G(y) = ∫_0^y √g(s) ds (g = ρ)."""
    ys, G = _sqrt_g_prefix(g_fn, n_grid=n_grid)
    G1 = float(G[-1])
    t = np.asarray(t, dtype=np.float64)
    if G1 <= 0.0:
        return t.copy()
    return np.interp(t * G1, G, ys)
