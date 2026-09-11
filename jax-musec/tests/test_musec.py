"""Equation-level tests for Musec (arXiv:2609.11655)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musec.clip import (
    clip_singular_values,
    hard_spectral_clip,
    soft_saturation,
    soft_spectral_clip,
    soft_spectral_clip_exact,
)
from musec.optimizer import init, step

jax.config.update("jax_enable_x64", True)


def _known_matrix():
    """M = U S V^T with known singular values 0.2, 0.8, 2.5."""
    U, _ = jnp.linalg.qr(jnp.array([[1.0, 2.0, 0.5], [0.0, 1.0, 1.5], [1.0, 0.0, 1.0]]))
    V, _ = jnp.linalg.qr(jnp.array([[1.0, 0.5, 0.2, 0.0], [0.0, 1.0, 0.3, 0.1], [0.2, 0.0, 1.0, 0.4], [0.0, 0.1, 0.0, 1.0]]))
    s = jnp.array([2.5, 0.8, 0.2])
    Vh = V[:, :3].T
    M = (U * s) @ Vh
    return M, U, s, Vh


def test_clip_leaves_small_singular_values_and_caps_large():
    # Section 3.1: clip(w, D) = min{w, D}; Algorithm 2 steps 11–13.
    M, U, s, Vh = _known_matrix()
    D = 1.0
    Mc = hard_spectral_clip(M, D)
    _, s_c, _ = jnp.linalg.svd(Mc, full_matrices=False)
    expected = jnp.minimum(s, D)
    np.testing.assert_allclose(np.sort(s_c)[::-1], expected, atol=1e-10)
    # Reconstruction U Clip(S, D) V^T.
    expected_M = (U * expected) @ Vh
    np.testing.assert_allclose(Mc, expected_M, atol=1e-10)


def test_clipped_spectral_norm_at_most_D():
    M, _, _, _ = _known_matrix()
    D = 0.5
    Mc = hard_spectral_clip(M, D)
    spec = jnp.linalg.norm(Mc, ord=2)
    np.testing.assert_allclose(float(spec), D, atol=1e-10)


def test_clip_is_identity_when_all_singular_values_below_D():
    M, _, s, _ = _known_matrix()
    D = float(s.max()) + 1.0
    Mc = hard_spectral_clip(M, D)
    np.testing.assert_allclose(Mc, M, atol=1e-10)


def test_hard_clip_becomes_scaled_polar_when_all_sv_exceed_D():
    # If every singular value > D, Clip(S, D) = D I and M maps to D U V^T.
    M, U, s, Vh = _known_matrix()
    D = 0.05
    assert float(s.min()) > D
    Mc = hard_spectral_clip(M, D)
    polar = D * (U @ Vh)
    np.testing.assert_allclose(Mc, polar, atol=1e-10)


def test_soft_saturation_limits():
    # Section 5: h(w, D) ≈ w for w << D and h(w, D) ≈ D for w >> D.
    D = 1.0
    small = jnp.array([1e-4, 1e-3, 0.01])
    large = jnp.array([1e2, 1e3, 1e4])
    np.testing.assert_allclose(soft_saturation(small, D), small, rtol=1e-4)
    np.testing.assert_allclose(soft_saturation(large, D), jnp.full_like(large, D), rtol=1e-4)
    # Always strictly below D for finite w > 0, and equal to clip at 0.
    mid = jnp.array([0.0, 0.5, 1.0, 2.0])
    h = soft_saturation(mid, D)
    assert float(h[1]) < D
    assert float(h[2]) < D
    assert float(h[3]) < D
    np.testing.assert_allclose(h[0], 0.0)


def test_soft_exact_matches_eq3_and_svd_map():
    # Eq. (3): H(M, D) = D (M M^T + D^2 I)^{-1/2} M
    M, _, s, _ = _known_matrix()
    D = 1.0
    H = soft_spectral_clip_exact(M, D)
    A = M @ M.T + D**2 * jnp.eye(M.shape[0])
    evals, evecs = jnp.linalg.eigh(A)
    inv_sqrt = evecs * (evals ** -0.5) @ evecs.T
    H_eq3 = D * (inv_sqrt @ M)
    np.testing.assert_allclose(H, H_eq3, atol=1e-10)
    _, s_h, _ = jnp.linalg.svd(H, full_matrices=False)
    np.testing.assert_allclose(np.sort(s_h)[::-1], soft_saturation(s, D), atol=1e-10)


def test_newton_schulz_matches_smooth_saturation():
    # Algorithm 3 approximates Eq. (3) on a small well-conditioned matrix.
    M, _, _, _ = _known_matrix()
    D = 1.0
    H_ns = soft_spectral_clip(M, D, ns_steps=8)
    H_ex = soft_spectral_clip_exact(M, D)
    np.testing.assert_allclose(H_ns, H_ex, atol=1e-8, rtol=1e-7)


def test_ssc_handles_tall_and_wide_matrices():
    key = jax.random.PRNGKey(0)
    D = 0.7
    wide = jax.random.normal(key, (3, 8))
    tall = wide.T
    H_wide = soft_spectral_clip(wide, D, ns_steps=8)
    H_tall = soft_spectral_clip(tall, D, ns_steps=8)
    np.testing.assert_allclose(H_wide, soft_spectral_clip_exact(wide, D), atol=1e-8)
    np.testing.assert_allclose(H_tall, soft_spectral_clip_exact(tall, D), atol=1e-8)
    np.testing.assert_allclose(H_tall, H_wide.T, atol=1e-8)


def test_clipped_momentum_is_fed_back_into_ema():
    # Section 4.1 / Appendix D: Mn (clipped) is the next momentum buffer.
    W = jnp.array([[3.0, 0.0], [0.0, 0.1]])
    g0 = jnp.array([[4.0, 0.0], [0.0, 0.2]])
    g1 = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    D, beta, eta = 1.0, 0.5, 0.1
    params = {"W": W}
    state = init(params)
    params, state = step(params, {"W": g0}, state, eta=eta, beta=beta, D=D, mode="hard")
    M0 = hard_spectral_clip(g0, D)
    np.testing.assert_allclose(state.momentum["W"], M0, atol=1e-10)
    # Second step mixes clipped M0, not the raw g0.
    params, state = step(params, {"W": g1}, state, eta=eta, beta=beta, D=D, mode="hard")
    mixed = (1.0 - beta) * M0 + beta * g1
    expected = hard_spectral_clip(mixed, D)
    np.testing.assert_allclose(state.momentum["W"], expected, atol=1e-10)
    # SPECTRA-style (unclipped buffer) would mix raw g0 instead.
    spectra_mixed = (1.0 - beta) * g0 + beta * g1
    assert not np.allclose(spectra_mixed, mixed)


def test_first_step_uses_raw_gradient_as_momentum():
    # Algorithm 2, n = 0: Mc_0 = G_0 (not beta * G_0).
    W = jnp.eye(2)
    g = jnp.array([[2.0, 0.0], [0.0, 0.3]])
    D, beta, eta = 1.0, 0.1, 0.05
    params = {"W": W}
    state = init(params)
    new_params, state = step(params, {"W": g}, state, eta=eta, beta=beta, D=D, mode="hard")
    Mc = hard_spectral_clip(g, D)
    np.testing.assert_allclose(new_params["W"], W - eta * Mc, atol=1e-12)
    assert int(state.step) == 1


def test_pytree_2d_clipped_1d_unclipped():
    params = {
        "W": jnp.array([[3.0, 0.0], [0.0, 0.4]]),
        "b": jnp.array([1.0, -2.0]),
        "s": jnp.array(0.5),
    }
    grads = {
        "W": jnp.array([[5.0, 0.0], [0.0, 0.2]]),
        "b": jnp.array([0.3, -0.1]),
        "s": jnp.array(0.25),
    }
    D, beta, eta = 1.0, 0.5, 0.1
    state = init(params)
    new_params, state = step(params, grads, state, eta=eta, beta=beta, D=D, mode="hard")
    Mc_W = hard_spectral_clip(grads["W"], D)
    np.testing.assert_allclose(new_params["W"], params["W"] - eta * Mc_W)
    np.testing.assert_allclose(new_params["b"], params["b"] - eta * grads["b"])
    np.testing.assert_allclose(new_params["s"], params["s"] - eta * grads["s"])
    np.testing.assert_allclose(state.momentum["b"], grads["b"])
    # 2D leaf is spectrally bounded; 1D is not passed through Clip_sc.
    spec = jnp.linalg.norm(state.momentum["W"], ord=2)
    np.testing.assert_allclose(float(spec), min(float(jnp.linalg.norm(grads["W"], ord=2)), D), atol=1e-10)


def test_one_step_decreases_quadratic_loss():
    key = jax.random.PRNGKey(1)
    W_star = jax.random.normal(key, (4, 3))
    W = W_star + 0.5 * jax.random.normal(jax.random.fold_in(key, 1), (4, 3))
    b = jnp.zeros(4)

    def loss(p):
        return 0.5 * jnp.sum((p["W"] - W_star) ** 2) + 0.5 * jnp.sum(p["b"] ** 2)

    params = {"W": W, "b": b}
    g = jax.grad(loss)(params)
    state = init(params)
    new_params, _ = step(params, g, state, eta=0.2, beta=0.5, D=1.0, mode="hard")
    assert np.isfinite(loss(new_params))
    assert float(loss(new_params)) < float(loss(params))


def test_soft_step_produces_finite_params():
    params = {"W": jnp.array([[1.0, 2.0], [0.5, -1.0], [0.0, 0.3]])}
    grads = {"W": jnp.array([[0.4, -0.2], [1.5, 0.1], [0.0, -0.8]])}
    state = init(params)
    new_params, new_state = step(
        params, grads, state, eta=0.05, beta=0.9, D=0.5, mode="soft", ns_steps=5
    )
    assert jnp.all(jnp.isfinite(new_params["W"]))
    assert jnp.all(jnp.isfinite(new_state.momentum["W"]))


def test_clip_singular_values_matches_section_31():
    s = jnp.array([0.0, 0.3, 1.0, 2.4])
    np.testing.assert_array_equal(clip_singular_values(s, 1.0), jnp.array([0.0, 0.3, 1.0, 1.0]))
