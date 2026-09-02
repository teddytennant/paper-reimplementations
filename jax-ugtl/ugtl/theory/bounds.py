"""Closed-form approximation ratios from Theorems 4, 5 and 6.

Every formula here is transcribed from the statement of the theorem it is named
after. The functions are plain floats, not jax arrays, because they are read off
symbolic expressions and the tests compare them against limits.
"""

from __future__ import annotations

import math

__all__ = [
    "alpha_greedy",
    "alpha_local_search",
    "alpha_pgd",
    "pgd_step_size",
    "greedy_value_oracle_ratio",
]


def _noise_ratio(beta: float) -> float:
    """The (1 - beta) / (1 + beta) factor that shows up in both value-query proofs."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must lie in [0, 1), got {beta}")
    return (1.0 - beta) / (1.0 + beta)


def greedy_value_oracle_ratio(gamma: float, k: int) -> float:
    """1 - (1 - gamma / k)^k, the greedy ratio under an exact value oracle.

    This is the beta = 0 specialization of Theorem 4, quoted in the paragraph
    that introduces the theorem.
    """
    return 1.0 - (1.0 - gamma / k) ** k


def alpha_greedy(beta: float, gamma: float, k: int) -> float:
    """Approximation ratio alpha of Theorem 4.

        alpha = [ r * gamma / k / (1 - r * (1 - gamma / k)) ]
                * [ 1 - r^k * (1 - gamma / k)^k ]

    with r = (1 - beta) / (1 + beta), gamma the submodularity ratio
    gamma_{X_k, k} of Definition 6 and k the size constraint of Definition 7.

    Limits the paper states, both checked in the tests:
      * beta = 0 gives 1 - (1 - gamma / k)^k, the value-oracle greedy ratio;
      * gamma = 1 and beta = 1 / n^2 gives 1 - 1/e - o(1).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    r = _noise_ratio(beta)
    g = gamma / k
    denom = 1.0 - r * (1.0 - g)
    return (r * g / denom) * (1.0 - (r * (1.0 - g)) ** k)


def alpha_local_search(beta: float, eps: float, n: int) -> float:
    """Approximation ratio alpha of Theorem 5.

        alpha = ((1 - beta) / (1 + beta))
                * (3 + 2n * (((1 + beta) / (1 - beta)) * (1 + eps / n^2) - 1))^-1

    eps is the slack parameter of Algorithm 2, which only accepts a move that
    improves the surrogate value by a factor larger than 1 + eps / n^2, and n is
    the size of the ground set. At beta = 0 this reduces to 1 / (3 + 2 eps / n).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    r = _noise_ratio(beta)
    inner = (1.0 / r) * (1.0 + eps / n**2) - 1.0
    return r / (3.0 + 2.0 * n * inner)


def pgd_step_size(zeta_g: float, diameter: float, grad_bound: float, T: int) -> float:
    """Step size eta = D / ((G + zeta_g) sqrt(T)) that Theorem 6 assumes."""
    if T < 1:
        raise ValueError(f"T must be at least 1, got {T}")
    return diameter / ((grad_bound + zeta_g) * math.sqrt(T))


def alpha_pgd(zeta_g: float, diameter: float, grad_bound: float, T: int) -> float:
    """Additive approximation alpha of Theorem 6.

        alpha = (G + zeta_g) D / sqrt(T) + zeta_g D

    D is the diameter of the convex domain, G bounds the true gradient norm on
    it, and zeta_g is the trajectory-wise gradient error of Definition 5. At
    zeta_g = 0 this is the textbook D G / sqrt(T) rate for Algorithm 3.
    """
    if T < 1:
        raise ValueError(f"T must be at least 1, got {T}")
    if zeta_g < 0.0:
        raise ValueError(f"zeta_g must be non-negative, got {zeta_g}")
    return (grad_bound + zeta_g) * diameter / math.sqrt(T) + zeta_g * diameter
