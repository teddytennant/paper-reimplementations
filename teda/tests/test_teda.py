"""Equation and invariant tests for TEDA (arXiv:2609.28967). CPU, pytest."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from teda.extract import (
    extract_from_radiance,
    extract_transmittance,
    gate,
    noise_scale_tau_g,
)
from teda.match import contrast_weights, match_distance, operator_consistent_transmittance, run_batch
from teda.operators import (
    ETA_HIGH,
    ETA_LOW,
    absorption_projector,
    branch_operators,
    build_stage1_operators,
    second_difference_operator,
)
from teda.planck import linear_in_T, planck_B
from teda.synth import (
    channel_kappa,
    channel_grid,
    fine_grid,
    observe_radiance,
    sensor_transmittance,
    synthetic_kappa,
    transmittance_bank,
)


def _scene(n_channels=64, n_fine=801, fwhm_um=None):
    lam_ch = channel_grid(n_channels)
    lam_fine = fine_grid(n_fine)
    kappa_fine = synthetic_kappa(lam_fine)
    alpha_tilde = channel_kappa(lam_fine, kappa_fine, lam_ch, fwhm_um=fwhm_um)
    ops = build_stage1_operators(alpha_tilde)
    return lam_ch, lam_fine, kappa_fine, alpha_tilde, ops


def test_planck_positive_and_increases_with_T():
    lam = jnp.linspace(8.0, 13.0, 16)
    b_cool = planck_B(lam, 280.0)
    b_warm = planck_B(lam, 310.0)
    assert jnp.all(b_cool > 0)
    assert jnp.all(b_warm > b_cool)


def test_M_and_H_annihilate_absorption_direction():
    """M alpha = 0 and H_eta alpha = 0 for both frozen eta, eqs (24), (26)."""
    _, _, _, _, ops = _scene()
    alpha = ops.alpha
    assert jnp.allclose(ops.M @ alpha, 0.0, atol=1e-12)
    H_low, _, _, _ = branch_operators(ops.M, ETA_LOW, D2=ops.D2)
    H_high, _, _, _ = branch_operators(ops.M, ETA_HIGH, D2=ops.D2)
    assert jnp.allclose(H_low @ alpha, 0.0, atol=1e-10)
    assert jnp.allclose(H_high @ alpha, 0.0, atol=1e-10)
    assert jnp.allclose(ops.H_low @ alpha, 0.0, atol=1e-10)
    assert jnp.allclose(ops.H_high @ alpha, 0.0, atol=1e-10)


def test_linear_continuum_recovers_unit_transmittance():
    """Degree <= 1 continuum with tau=1 is eaten by the smooth baseline: tau_hat ~ 1."""
    lam_ch, _, _, _, ops = _scene()
    z = 0.4 + 0.05 * (lam_ch - 10.5)
    extracted = extract_transmittance(z, ops, tau_g=1.0)
    assert jnp.allclose(extracted.tau_hat, 1.0, atol=1e-6, rtol=1e-6)


def test_closed_form_matches_explicit_solve():
    """H_eta z equals the solve of (M + eta D2^T D2) b = M z, eq (26)."""
    lam_ch, _, _, _, ops = _scene()
    z = jnp.sin(0.8 * (lam_ch - 8.0)) + 0.1 * lam_ch
    for eta, H in ((ops.eta_low, ops.H_low), (ops.eta_high, ops.H_high)):
        D2 = second_difference_operator(z.shape[0])
        A = ops.M + eta * (D2.T @ D2)
        b_solve = jnp.linalg.solve(A, ops.M @ z)
        assert jnp.allclose(H @ z, b_solve, atol=1e-10, rtol=1e-10)


def test_gate_limits():
    """g in (0, 1], g -> 1 as Delta -> 0, g -> 0 as |Delta| -> inf, eq (29)."""
    tau_g = 0.05
    g0 = gate(0.0, tau_g)
    g_mid = gate(tau_g, tau_g)
    g_far = gate(1e6, tau_g)
    g_neg = gate(-1e6, tau_g)
    assert float(g0) == pytest.approx(1.0)
    assert 0.0 < float(g_mid) < 1.0
    assert 0.0 < float(g_far) <= 1.0
    assert float(g_far) < 1e-20
    assert float(g_neg) < 1e-20
    deltas = jnp.array([-2.0, -0.1, 0.0, 0.1, 3.5])
    gs = gate(deltas, tau_g)
    assert jnp.all(gs > 0.0)
    assert jnp.all(gs <= 1.0)


def test_weights_nonnegative_and_normalized():
    b_hat = jnp.array([-1.2, 0.0, 0.4, 1.1, -0.3])
    w = contrast_weights(b_hat)
    assert jnp.all(w >= 0.0)
    assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-12)


def test_noise_free_ranging_within_one_grid_step():
    """Noise-free, no-ISRF, eps=1, Delta T != 0: recover d* on a coarse grid."""
    lam_ch, lam_fine, kappa_fine, _, ops = _scene(n_channels=48)
    d_grid = jnp.array([50.0, 100.0, 150.0, 200.0])
    tau_bank = transmittance_bank(d_grid, lam_fine, kappa_fine, lam_ch, fwhm_um=None)
    T_air = 290.0
    T_obj = 305.0
    B_air = linear_in_T(T_air, scale=1.0, like=lam_ch)
    B_obj = linear_in_T(T_obj, scale=1.0, like=lam_ch)
    for d_true in d_grid:
        tau = sensor_transmittance(d_true, lam_fine, kappa_fine, lam_ch, fwhm_um=None)
        y = observe_radiance(tau, B_obj, B_air, emissivity=1.0)
        extracted = extract_from_radiance(y, B_air, ops, sigma=0.0)
        ln_bank = jnp.log(tau_bank)
        tau_tilde = operator_consistent_transmittance(ln_bank, extracted.P_g)
        w = contrast_weights(extracted.b_hat)
        matched = match_distance(extracted.tau_hat, tau_tilde, w, d_grid)
        assert float(matched.d_hat) == pytest.approx(float(d_true), abs=1e-9)


def test_isrf_bank_beats_distance_invariant_kappa_eff():
    """ISRF-integrated tau: operator-consistent bank vs exp(-kappa_eff d).

    Strong check: matching the true ISRF tau at d* against the two model
    families, the per-distance ISRF bank is closer to d* than a fixed
    kappa_eff (Jensen / distance-invariant coefficient, the paper's bias
    source). Weak invariant always required: operator-consistent cost at
    the true d is smaller than at a far-away d.
    """
    fwhm = 0.08
    lam_ch, lam_fine, kappa_fine, alpha_tilde, ops = _scene(
        n_channels=41, n_fine=1201, fwhm_um=fwhm
    )
    d_grid = jnp.array([50.0, 100.0, 150.0, 200.0])
    d_true = 100.0
    tau_bank = transmittance_bank(d_grid, lam_fine, kappa_fine, lam_ch, fwhm_um=fwhm)
    tau_true = sensor_transmittance(d_true, lam_fine, kappa_fine, lam_ch, fwhm_um=fwhm)

    T_air = 290.0
    T_obj = 308.0
    B_air = linear_in_T(T_air, scale=1.0, like=lam_ch)
    B_obj = linear_in_T(T_obj, scale=1.0, like=lam_ch)
    y = observe_radiance(tau_true, B_obj, B_air, emissivity=1.0)
    extracted = extract_from_radiance(y, B_air, ops, sigma=0.0)
    w = contrast_weights(extracted.b_hat)

    ln_bank = jnp.log(jnp.maximum(tau_bank, 1e-30))
    tau_tilde = operator_consistent_transmittance(ln_bank, extracted.P_g)
    matched_bank = match_distance(extracted.tau_hat, tau_tilde, w, d_grid)

    # Distance-invariant kappa_eff: channel-integrated kappa, Jensen model.
    kappa_eff = alpha_tilde
    tau_keff = jnp.exp(-kappa_eff[None, :] * d_grid[:, None])
    ln_keff = jnp.log(jnp.maximum(tau_keff, 1e-30))
    tau_keff_op = operator_consistent_transmittance(ln_keff, extracted.P_g)
    matched_keff = match_distance(extracted.tau_hat, tau_keff_op, w, d_grid)

    err_bank = abs(float(matched_bank.d_hat) - d_true)
    err_keff = abs(float(matched_keff.d_hat) - d_true)
    assert err_bank <= err_keff

    idx_true = int(jnp.argmin(jnp.abs(d_grid - d_true)))
    idx_far = int(jnp.argmin(jnp.abs(d_grid - 200.0)))
    assert float(matched_bank.cost[idx_true]) < float(matched_bank.cost[idx_far])


def test_batch_vmap_over_pixels():
    """Eight pixels through jax.vmap, no Python loop over pixels in the hot path."""
    lam_ch, lam_fine, kappa_fine, _, ops = _scene(n_channels=48)
    d_grid = jnp.array([50.0, 100.0, 150.0, 200.0])
    tau_bank = transmittance_bank(d_grid, lam_fine, kappa_fine, lam_ch, fwhm_um=None)
    T_air = 290.0
    B_air = linear_in_T(T_air, scale=1.0, like=lam_ch)
    true_ds = jnp.array([50.0, 50.0, 100.0, 100.0, 150.0, 150.0, 200.0, 200.0])
    temps = jnp.array([300.0, 305.0, 302.0, 310.0, 298.0, 307.0, 301.0, 312.0])

    def _synth(d, T):
        tau = sensor_transmittance(d, lam_fine, kappa_fine, lam_ch, fwhm_um=None)
        B_obj = linear_in_T(T, scale=1.0, like=lam_ch)
        return observe_radiance(tau, B_obj, B_air, emissivity=1.0)

    y_batch = jax.vmap(_synth)(true_ds, temps)

    @jax.jit
    def _run(y):
        return run_batch(y, B_air, ops, tau_bank, d_grid, 0.0)

    out = _run(y_batch)
    assert out.d_hat.shape == (8,)
    assert out.tau_hat.shape == (8, lam_ch.shape[0])
    assert out.weights.shape == (8, lam_ch.shape[0])
    assert jnp.allclose(out.d_hat, true_ds, atol=1e-9)
    assert jnp.allclose(jnp.sum(out.weights, axis=-1), 1.0, atol=1e-12)


def test_absorption_projector_formula():
    alpha = jnp.array([0.2, -0.5, 0.1, 0.8])
    alpha = alpha / jnp.linalg.norm(alpha)
    M = absorption_projector(alpha)
    n = alpha.shape[0]
    assert jnp.allclose(M, jnp.eye(n) - jnp.outer(alpha, alpha), atol=1e-15)
    assert jnp.allclose(M @ alpha, 0.0, atol=1e-15)


def test_tau_g_positive():
    Y = jnp.array([0.2, -0.4, 0.15, 0.9])
    t0 = noise_scale_tau_g(Y, 0.0)
    t1 = noise_scale_tau_g(Y, 0.01)
    assert float(t0) > 0.0
    assert float(t1) > 0.0
