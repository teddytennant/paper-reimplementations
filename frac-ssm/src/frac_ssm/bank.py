"""Geometric timescale bank, exact ZOH, and Mittag-Leffler helpers.

Theorem 2: τ_m = τ0 * q^{m-1} for m = 1..M, q > 1.
Eq. (17-18): ρ = exp(-Δ/τ), β = (1-ρ)/λ  (theory ZOH).
Algorithm 1 / Remark 7: implemented injection is β̃ = 1-ρ̃.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def geometric_timescales(tau0, q, n_modes):
    """τ_m = τ0 * q^{m-1}, m = 1..M (Theorem 2)."""
    tau0 = jnp.asarray(tau0)
    q = jnp.asarray(q)
    m = jnp.arange(n_modes)
    return tau0 * q**m


def zoh_rho_beta(delta, tau, lam):
    """Exact ZOH of ṡ = -s/τ + u/(λ τ).

    ρ = exp(-Δ/τ), β = (1-ρ)/λ  (eq. 17-18).
    """
    delta = jnp.asarray(delta)
    tau = jnp.asarray(tau)
    lam = jnp.asarray(lam)
    rho = jnp.exp(-delta / tau)
    beta = (1.0 - rho) / lam
    return rho, beta


def implemented_rho_beta(delta, tau_tilde):
    """Algorithm 1 step 3: ρ̃ = exp(-Δ/τ̃), β̃ = 1-ρ̃ (not (1-ρ̃)/λ)."""
    rho = jnp.exp(-delta / tau_tilde)
    beta = 1.0 - rho
    return rho, beta


def effective_timescales(tau, lam, alpha):
    """τ̃ = τ / λ^{1/α}  (eq. 19)."""
    lam = jnp.clip(jnp.asarray(lam), 1e-4, 1e3)
    alpha = jnp.clip(jnp.asarray(alpha), 0.05, 0.99)
    # λ^{1/α} = exp(log λ / α), clipped so τ̃ stays in a usable range.
    scale = jnp.exp(jnp.clip(jnp.log(lam) / alpha, -20.0, 20.0))
    return jnp.clip(jnp.asarray(tau) / scale, 1e-4, 1e4)


def zoh_closed_form(s0, u, delta, tau, a):
    """Exact solution of ṡ = -s/τ + a u with constant u on [0, Δ].

    s(Δ) = e^{-Δ/τ} s0 + (1 - e^{-Δ/τ}) a τ u.
    """
    rho = jnp.exp(-delta / tau)
    return rho * s0 + (1.0 - rho) * a * tau * u


def mittag_leffler_series(z, alpha, n_terms=80):
    """E_α(z) = Σ_{k=0}^{n-1} z^k / Γ(α k + 1)."""
    z = jnp.asarray(z)
    k = jnp.arange(n_terms)
    terms = jnp.power(z[..., None], k) / jax.scipy.special.gamma(alpha * k + 1.0)
    return jnp.sum(terms, axis=-1)


def mittag_leffler_e_alpha(t, alpha, lam=1.0, n_terms=80):
    """E_α(-λ t^α) via power series."""
    t = jnp.asarray(t)
    z = -lam * jnp.power(t, alpha)
    return mittag_leffler_series(z, alpha, n_terms=n_terms)


def ml_spectral_density(xi, alpha, lam):
    """Bernstein density K(ξ) for E_α(-λ t^α) = ∫ e^{-ξ t} K(ξ) dξ."""
    xi = jnp.asarray(xi)
    sap = jnp.power(xi, alpha)
    num = (lam / jnp.pi) * jnp.power(xi, alpha - 1.0) * jnp.sin(jnp.pi * alpha)
    den = sap**2 + 2.0 * lam * sap * jnp.cos(jnp.pi * alpha) + lam**2
    return num / den


def soe_coeffs_spectral(tau, alpha, lam, q):
    """Geometric quadrature weights for the ML spectral measure.

    ξ_m = 1/τ_m, c_m = K(ξ_m) ξ_m log(q), so Σ_m c_m e^{-t/τ_m} ≈ E_α(-λ t^α).
    The large-τ limit of this formula is Theorem 2 eq. (12).
    """
    tau = jnp.asarray(tau)
    xi = 1.0 / tau
    kdens = ml_spectral_density(xi, alpha, lam)
    return kdens * xi * jnp.log(q)


def soe_coeffs_leading(tau, alpha, lam, q):
    """Leading large-timescale term (12): c_m = ((log q) sin(πα)/(π λ)) τ_m^{-α}."""
    tau = jnp.asarray(tau)
    return ((jnp.log(q) * jnp.sin(jnp.pi * alpha)) / (jnp.pi * lam)) * jnp.power(tau, -alpha)


def soe_eval(t, tau, coeffs):
    """Σ_m c_m exp(-t / τ_m)."""
    t = jnp.asarray(t)
    tau = jnp.asarray(tau)
    coeffs = jnp.asarray(coeffs)
    return jnp.sum(coeffs * jnp.exp(-t[..., None] / tau), axis=-1)
