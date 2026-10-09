"""Kish effective sample size, section 3.4.

    Neff = 1 / ((1 - lam)^2 / N_T + lam^2 / N_S)

When lam = N_S / (N_T + N_S), Neff = N_T + N_S.
"""

from __future__ import annotations


def kish_neff(lam: float, n_target: float, n_source: float) -> float:
    lam = float(lam)
    n_target = float(n_target)
    n_source = float(n_source)
    if n_target <= 0 or n_source <= 0:
        raise ValueError("N_T and N_S must be positive")
    return 1.0 / ((1.0 - lam) ** 2 / n_target + lam**2 / n_source)
