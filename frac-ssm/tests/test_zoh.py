"""Geometric bank, exact ZOH (17-18), Algorithm 1 β̃, Mittag-Leffler SoE."""

import jax.numpy as jnp

from frac_ssm.bank import (
    effective_timescales,
    geometric_timescales,
    implemented_rho_beta,
    mittag_leffler_e_alpha,
    soe_coeffs_spectral,
    soe_eval,
    zoh_closed_form,
    zoh_rho_beta,
)
from frac_ssm.transition import alg1_rho_beta_kappa


def test_geometric_bank_theorem2():
    tau0, q, M = 0.25, 3.0, 5
    tau = geometric_timescales(tau0, q, M)
    expected = jnp.array([tau0 * q ** (m - 1) for m in range(1, M + 1)])
    assert tau.shape == (M,)
    assert jnp.allclose(tau, expected)


def test_zoh_matches_scalar_ode():
    s0, u, delta, tau, lam = 0.4, 1.3, 0.7, 2.5, 0.8
    a = 1.0 / (lam * tau)
    closed = zoh_closed_form(s0, u, delta, tau, a)
    rho, beta = zoh_rho_beta(delta, tau, lam)
    stepped = rho * s0 + beta * u
    analytic = jnp.exp(-delta / tau) * s0 + (1.0 - jnp.exp(-delta / tau)) * a * tau * u
    assert jnp.allclose(closed, analytic)
    assert jnp.allclose(stepped, analytic)
    assert jnp.allclose(beta, (1.0 - rho) / lam)


def test_algorithm1_beta_is_one_minus_rho():
    tau = geometric_timescales(0.5, 2.0, 4)
    delta = jnp.array(0.3)
    alpha = jnp.array(0.6)
    lam = jnp.array(1.7)
    tau_tilde = effective_timescales(tau, lam, alpha)
    rho, beta = implemented_rho_beta(delta, tau_tilde)
    assert jnp.allclose(beta, 1.0 - rho)
    theory_beta = (1.0 - rho) / lam
    assert not jnp.allclose(beta, theory_beta)

    g = jnp.zeros((1, 2, 4))
    delta_h = jnp.full((1, 2), 0.3)
    alpha_h = jnp.full((1, 2), 0.6)
    lam_h = jnp.full((1, 2), 1.7)
    rho2, beta2, kappa, c, b = alg1_rho_beta_kappa(
        delta_h, alpha_h, lam_h, tau, g, g
    )
    assert jnp.allclose(beta2, 1.0 - rho2)
    assert kappa.shape == b.shape


def test_soe_error_drops_with_more_modes():
    alpha, lam = 0.5, 1.0
    t = jnp.linspace(0.0, 1.0, 64)
    target = mittag_leffler_e_alpha(t, alpha, lam=lam, n_terms=80)
    tau_min, tau_max = 1e-3, 40.0

    def err(M):
        q = (tau_max / tau_min) ** (1.0 / (M - 1))
        tau = geometric_timescales(tau_min, q, M)
        c = soe_coeffs_spectral(tau, alpha, lam, q)
        approx = soe_eval(t, tau, c)
        return float(jnp.mean(jnp.square(approx - target)))

    assert err(8) < err(2)
