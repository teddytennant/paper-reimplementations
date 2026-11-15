"""Lemma 2 functionals of ρ."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tauleap_schedule.quad import as_numpy_fn, trapz


def D(rho_fn: Callable, n_quad: int = 2048) -> float:
    """D = ∫_0^1 ρ(u) du."""
    rho = as_numpy_fn(rho_fn)
    xs = np.linspace(0.0, 1.0, n_quad)
    return trapz(rho(xs), xs)


def TC(rho_fn: Callable, N: int, n_quad: int = 2048) -> float:
    """TC = N ∫_0^1 (1 - u) ρ(u) du."""
    rho = as_numpy_fn(rho_fn)
    xs = np.linspace(0.0, 1.0, n_quad)
    return float(N) * trapz((1.0 - xs) * rho(xs), xs)


def DTC(rho_fn: Callable, N: int, n_quad: int = 2048) -> float:
    """DTC = N ∫_0^1 u ρ(u) du."""
    rho = as_numpy_fn(rho_fn)
    xs = np.linspace(0.0, 1.0, n_quad)
    return float(N) * trapz(xs * rho(xs), xs)
