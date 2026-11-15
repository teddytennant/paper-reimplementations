"""Factorization error ε_fact(β) (Theorem 1)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tauleap_schedule.quad import as_numpy_fn, trapz


def eps_fact(beta, rho_fn: Callable, N: int, n_quad: int = 1024) -> float:
    """ε_fact(β) = N Σ_k ∫_{β_{k-1}}^{β_k} (β_k - u) ρ(u) du.

    ``beta`` is the knot vector 0 = β_0 < ⋯ < β_K = 1.
    """
    beta = np.asarray(beta, dtype=np.float64)
    rho = as_numpy_fn(rho_fn)
    total = 0.0
    for k in range(1, beta.size):
        a = float(beta[k - 1])
        b = float(beta[k])
        if b <= a:
            continue
        xs = np.linspace(a, b, n_quad)
        integrand = (b - xs) * rho(xs)
        total += trapz(integrand, xs)
    return float(N * total)
