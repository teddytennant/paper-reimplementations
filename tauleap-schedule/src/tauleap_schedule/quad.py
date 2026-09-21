"""Tiny quadrature helpers (numpy; JAX is not required for 1-D integrals)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def trapz(y, x) -> float:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def as_numpy_fn(fn: Callable) -> Callable:
    def wrapped(x):
        x_np = np.asarray(x, dtype=np.float64)
        out = fn(x_np)
        if hasattr(out, "__array__"):
            out = np.asarray(out)
        return np.asarray(out, dtype=np.float64).reshape(np.shape(x_np))

    return wrapped


def integrate(fn: Callable, a: float, b: float, n: int = 1024) -> float:
    if b <= a:
        return 0.0
    rho = as_numpy_fn(fn)
    xs = np.linspace(float(a), float(b), n)
    return trapz(rho(xs), xs)
