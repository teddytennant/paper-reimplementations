"""Brute-force enumeration of all nonempty supports for (P1)."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from robust_sparse_mv.closed_form import Xi, support_objective
from robust_sparse_mv.problem import RobustSparseMV


@dataclass(frozen=True)
class BruteResult:
    x: Array
    f_beta: float
    support: tuple[int, ...]
    n_xi_evals: int


def _mask_to_support(mask: int, n: int) -> tuple[int, ...]:
    return tuple(i for i in range(n) if mask & (1 << i))


def brute_force_p1(problem: RobustSparseMV) -> BruteResult:
    """Evaluate ``ξ(ω)`` on every nonempty ``ω ⊆ {0,...,N-1}`` and return the best ``F_β``."""
    n = problem.n
    best_val = float("inf")
    best_support: tuple[int, ...] = ()
    n_eval = 0
    for mask in range(1, 1 << n):
        omega = _mask_to_support(mask, n)
        val = float(support_objective(problem, omega))
        n_eval += 1
        if val < best_val - 1e-14 or (
            abs(val - best_val) <= 1e-14 and (best_support == () or omega < best_support)
        ):
            best_val = val
            best_support = omega
    x = Xi(problem, best_support)
    return BruteResult(x=x, f_beta=best_val, support=best_support, n_xi_evals=n_eval)
