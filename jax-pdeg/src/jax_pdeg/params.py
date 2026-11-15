"""Parameter choice (4.3) and warm-up length (4.31)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PDEGparams:
    """Algorithm 1/2 scalars.

    Attributes
    ----------
    L, mu, tau :
        Smoothness, dual strong-concavity, and dual regularization.
    lam :
        Primal proximal weight λ. Theorems use λ = 2L.
    h, alpha, gamma, beta :
        Step size, dual damping, dual momentum scale, center step.
    L_G, mu_y :
        L_G = 3L + τ and μ_y = μ + τ.
    """

    L: float
    mu: float
    tau: float
    lam: float
    h: float
    alpha: float
    gamma: float
    beta: float
    L_G: float
    mu_y: float

    @property
    def one_minus_alpha_gamma(self) -> float:
        return (1.0 - self.alpha) * self.gamma


def params_from_L_mu_tau(
    L: float,
    mu: float,
    tau: float,
    lam: float | None = None,
) -> PDEGparams:
    """Default parameters from (4.3).

    Computes h, then α from
        α = (1 + h √(2 L μ_y) / 16)^{-1},
    then γ from the (1-α)γ identity
        (1-α)γ = h (8L - √(2 L μ_y)) / (16 + h √(2 L μ_y)).
    This pair is consistent with γ = 4 √(2L / μ_y) - 1 in float64.
    """
    L = float(L)
    mu = float(mu)
    tau = float(tau)
    if L <= 0:
        raise ValueError("L must be positive")
    mu_y = mu + tau
    if mu_y <= 0:
        raise ValueError("mu + tau must be positive")
    if lam is None:
        lam = 2.0 * L
    else:
        lam = float(lam)
    L_G = 3.0 * L + tau
    h = 1.0 / (64.0 * L_G)
    s = math.sqrt(2.0 * L * mu_y)
    alpha = 1.0 / (1.0 + h * s / 16.0)
    one_m_alpha = 1.0 - alpha
    omag = h * (8.0 * L - s) / (16.0 + h * s)
    gamma = omag / one_m_alpha
    beta = h * s / 4096.0
    return PDEGparams(
        L=L,
        mu=mu,
        tau=tau,
        lam=lam,
        h=h,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        L_G=L_G,
        mu_y=mu_y,
    )


pdeg_params = params_from_L_mu_tau


def warmup_length(beta: float, H_mu_tau: float, eta: float) -> int:
    """T_w from (4.31) with the contraction (C17): (1 + 128 β) H_{t+1} ≤ H_t.

    q = max(1, H^{μ,τ} / η),  T_w = ceil( log(q) / log(1 + 128 β) ).
    Returns 0 when H^{μ,τ} ≤ η.
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be positive to compute T_w")
    q = max(1.0, float(H_mu_tau) / float(eta))
    if q <= 1.0:
        return 0
    return int(math.ceil(math.log(q) / math.log(1.0 + 128.0 * beta)))


def init_energy_bound(grad_x0, grad_y0, y0, L: float, mu: float, tau: float, D_Y: float) -> float:
    """Upper bound H^{μ,τ} on the initial Lyapunov value (Lemma C.1 / (4.30)).

    H^{μ,τ} = D_Y ||∇_y f(x0,y0)||
            + (||∇_x f||^2 + ||-∇_y f + τ y0||^2) / L
            + (τ/2 + (μ+τ)/256) D_Y^2.
    """
    gx = np.asarray(grad_x0, dtype=np.float64).reshape(-1)
    gy = np.asarray(grad_y0, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    mu_y = float(mu) + float(tau)
    r = -gy + float(tau) * y0
    return float(
        float(D_Y) * np.linalg.norm(gy)
        + (np.dot(gx, gx) + np.dot(r, r)) / float(L)
        + (float(tau) / 2.0 + mu_y / 256.0) * float(D_Y) ** 2
    )
