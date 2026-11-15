"""Chebyshev–Lobatto nodes and integrated Lagrange weights (Appendix B.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChebyshevLobatto:
    """Nodes t_j on [0, h] and integrated weights ω, ω', ω_end."""

    t: np.ndarray  # (J,)
    omega: np.ndarray  # (J, J)  ω_{i,j} = ∫_0^{t_i} (t_i - t) ℓ_j(t) dt
    omega_prime: np.ndarray  # (J, J)  ω'_{i,j} = ∫_0^{t_i} ℓ_j(t) dt
    omega_end: np.ndarray  # (J,)  ω_j = ∫_0^h ℓ_j(t) dt


def chebyshev_lobatto_nodes(J: int, h: float) -> np.ndarray:
    """Nodes t_j = (h/2) (1 - cos(π (j-1)/(J-1))) for j = 1..J."""
    if J < 2:
        raise ValueError("J must be >= 2")
    if h <= 0:
        raise ValueError("h must be positive")
    j = np.arange(1, J + 1, dtype=np.float64)
    return 0.5 * h * (1.0 - np.cos(np.pi * (j - 1) / (J - 1)))


def _lagrange_coeffs(j: int, nodes: np.ndarray) -> np.ndarray:
    """np.poly coefficients (highest degree first) of ℓ_j."""
    coeff = np.array([1.0], dtype=np.float64)
    tj = nodes[j]
    for k, tk in enumerate(nodes):
        if k != j:
            coeff = np.polymul(coeff, np.array([1.0, -float(tk)], dtype=np.float64))
    denom = np.polyval(coeff, tj)
    return coeff / denom


def _integrate_poly(coeff: np.ndarray, a: float, b: float) -> float:
    """Exact ∫_a^b p(t) dt for a polynomial given as np.poly coefficients."""
    n = coeff.shape[0]
    powers = np.arange(n, 0, -1, dtype=np.float64)
    anti = np.append(coeff / powers, 0.0)
    return float(np.polyval(anti, b) - np.polyval(anti, a))


def quadrature_weights(J: int, h: float) -> ChebyshevLobatto:
    """Nodes and weights (3.5)–(3.6) / Appendix B.1.

    ω_{i,j} = ∫_0^{t_i} (t_i - t) ℓ_j(t) dt
    ω'_{i,j} = ∫_0^{t_i} ℓ_j(t) dt
    ω_j = ∫_0^h ℓ_j(t) dt
    """
    t = chebyshev_lobatto_nodes(J, h)
    omega = np.zeros((J, J), dtype=np.float64)
    omega_prime = np.zeros((J, J), dtype=np.float64)
    omega_end = np.zeros(J, dtype=np.float64)
    for j in range(J):
        ell = _lagrange_coeffs(j, t)
        omega_end[j] = _integrate_poly(ell, 0.0, float(h))
        ell_times_t = np.polymul(ell, np.array([1.0, 0.0], dtype=np.float64))
        for i in range(J):
            ti = float(t[i])
            omega_prime[i, j] = _integrate_poly(ell, 0.0, ti)
            omega[i, j] = ti * omega_prime[i, j] - _integrate_poly(ell_times_t, 0.0, ti)
    return ChebyshevLobatto(t=t, omega=omega, omega_prime=omega_prime, omega_end=omega_end)
