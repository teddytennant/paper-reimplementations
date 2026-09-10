"""Paper claims for Muon-C (arXiv:2609.09676), CPU-only."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from muon_c import (
    conv_norm_grid,
    critical_fft,
    critical_ifft,
    lebesgue_constant,
    muon_c_direction,
    muon_c_scale,
    muon_c_step,
    patch_norm,
    polar_ns,
    polar_svd,
    rms,
    sampled_norm,
    spatial_norm,
    spatial_polar_direction,
    unfolded_muon_direction,
    uses_muon_c,
)

jax.config.update("jax_platforms", "cpu")


def _cosine(a, b):
    a = np.asarray(a).ravel().astype(np.complex128)
    b = np.asarray(b).ravel().astype(np.complex128)
    num = np.vdot(a, b).real
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return num / den


def test_polar_is_partial_isometry():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    tall = jax.random.normal(k1, (6, 3))
    fat = jax.random.normal(k2, (3, 6))
    square = jax.random.normal(k3, (4, 4))
    for M, fat_matrix in ((tall, False), (fat, True), (square, False)):
        P = polar_svd(M)
        gram = P @ P.T if fat_matrix else P.T @ P
        gram_np = np.asarray(gram)
        # A projector is Hermitian and idempotent.
        np.testing.assert_allclose(gram_np, gram_np.T, atol=1e-5)
        np.testing.assert_allclose(gram_np @ gram_np, gram_np, atol=1e-5)
        s = np.linalg.svd(np.asarray(P), compute_uv=False)
        assert np.all((s < 1e-5) | (np.abs(s - 1.0) < 1e-4))
    z = polar_svd(jnp.zeros((3, 2)))
    np.testing.assert_allclose(np.asarray(z), 0.0, atol=1e-7)


def test_proposition1_unitary_equivariance():
    key = jax.random.PRNGKey(1)
    kL, kR, kX = jax.random.split(key, 3)
    L, _ = jnp.linalg.qr(jax.random.normal(kL, (5, 5)))
    R, _ = jnp.linalg.qr(jax.random.normal(kR, (3, 3)))
    X = jax.random.normal(kX, (5, 3))
    left = polar_svd(L @ X @ R)
    right = L @ polar_svd(X) @ R
    np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-5)


def test_proposition2_sampled_norm_lmo():
    key = jax.random.PRNGKey(2)
    M = jax.random.normal(key, (3, 2, 3, 3))
    rho = 1.0
    n = 9.0
    O = muon_c_direction(M, polar="exact")
    U_star = -(rho / np.sqrt(n)) * np.asarray(O)
    samp = float(sampled_norm(jnp.asarray(U_star)))
    assert samp == pytest.approx(rho, rel=1e-4, abs=1e-4)

    kU = jax.random.PRNGKey(3)
    U_rand = jax.random.normal(kU, M.shape)
    scale = rho / (float(sampled_norm(U_rand)) + 1e-8)
    U_feas = np.asarray(U_rand) * scale
    assert float(sampled_norm(jnp.asarray(U_feas))) <= rho + 1e-4

    inner_star = float(np.sum(np.asarray(M) * U_star))
    inner_rand = float(np.sum(np.asarray(M) * U_feas))
    assert inner_star < inner_rand


def test_theorem3_norm_chain_and_lebesgue():
    key = jax.random.PRNGKey(4)
    U = jax.random.normal(key, (3, 2, 3, 3))
    sp = float(spatial_norm(U))
    patch = float(patch_norm(U))
    samp = float(sampled_norm(U))
    # 96 includes the 3x3 critical frequencies, so the grid max is a valid
    # lower bound on ||U||_conv and stays at least ||U||_samp.
    conv = float(conv_norm_grid(U, n_grid=96))
    assert sp <= patch + 1e-5
    assert patch <= samp + 1e-4
    assert samp <= conv + 1e-4

    lam3 = float(lebesgue_constant(3))
    assert lam3 == pytest.approx(5.0 / 3.0, rel=1e-3, abs=1e-3)
    assert lam3 <= np.sqrt(3.0) + 1e-6
    kappa_f = (5.0 / 3.0) ** 2
    assert kappa_f == pytest.approx(25.0 / 9.0)
    assert 9.0 / 25.0 > 1.0 / 3.0 > 1.0 / 9.0


def test_finite_support_identity():
    key = jax.random.PRNGKey(5)
    M = jax.random.normal(key, (4, 3, 3, 3))
    O = muon_c_direction(M, polar="exact")
    assert O.shape == M.shape
    Mhat = critical_fft(M)
    blocks = jnp.moveaxis(Mhat, (-4, -3), (-2, -1))
    Ohat = polar_svd(blocks)
    rec = critical_ifft(jnp.moveaxis(Ohat, (-2, -1), (-4, -3)))
    np.testing.assert_allclose(np.asarray(O), np.asarray(rec).real, atol=1e-5)
    assert rec.shape[-2:] == (3, 3)
    np.testing.assert_allclose(np.asarray(rec).imag, 0.0, atol=1e-5)


def test_real_momentum_stays_real():
    key = jax.random.PRNGKey(6)
    M = jax.random.normal(key, (2, 2, 3, 5))
    O_exact = muon_c_direction(M, polar="exact")
    O_ns = muon_c_direction(M, polar="ns")
    assert np.isrealobj(np.asarray(O_exact))
    assert np.isrealobj(np.asarray(O_ns))
    Mhat = critical_fft(M)
    rec_exact = critical_ifft(
        jnp.moveaxis(polar_svd(jnp.moveaxis(Mhat, (-4, -3), (-2, -1))), (-2, -1), (-4, -3))
    )
    rec_ns = critical_ifft(
        jnp.moveaxis(polar_ns(jnp.moveaxis(Mhat, (-4, -3), (-2, -1))), (-2, -1), (-4, -3))
    )
    np.testing.assert_allclose(np.asarray(rec_exact).imag, 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(rec_ns).imag, 0.0, atol=5e-5)


def test_newton_schulz_close_to_svd():
    # Five Frobenius-normalized NS steps (Appendix B) approximate the polar
    # factor. On a near-isometry the singular values move toward 1; do not
    # require machine-eps agreement with the SVD polar.
    key = jax.random.PRNGKey(7)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (8, 8)))
    B = Q
    P_ex = np.asarray(polar_svd(B))
    P_ns = np.asarray(polar_ns(B))
    rel = np.linalg.norm(P_ns - P_ex) / np.linalg.norm(P_ex)
    assert rel < 0.1
    raw = np.asarray(B / (np.linalg.norm(B) + 1e-8))
    sv_ns = np.linalg.svd(P_ns, compute_uv=False)
    sv_raw = np.linalg.svd(raw, compute_uv=False)
    assert np.mean(np.abs(sv_ns - 1.0)) < np.mean(np.abs(sv_raw - 1.0))


def test_scale_rms_is_point_two():
    key = jax.random.PRNGKey(8)
    M = jax.random.normal(key, (4, 3, 3, 3))
    O = muon_c_direction(M, polar="exact")
    s_f = float(muon_c_scale(O))
    applied = s_f * np.asarray(O)
    assert float(rms(jnp.asarray(applied))) == pytest.approx(0.2, rel=1e-5, abs=1e-5)


def test_block_partitions_are_not_collinear():
    # Figure 3: 1-channel 1x2 momentum M = (1, 2).
    M12 = jnp.array([1.0, 2.0]).reshape(1, 1, 1, 2)
    o_c = np.asarray(muon_c_direction(M12, polar="exact")).ravel()
    o_g = np.asarray(unfolded_muon_direction(M12, polar="exact")).ravel()
    o_s = np.asarray(spatial_polar_direction(M12)).ravel()
    np.testing.assert_allclose(o_c, np.array([0.0, np.sqrt(2.0)]), atol=1e-5)
    np.testing.assert_allclose(o_g, np.array([1.0, 2.0]) / np.sqrt(5.0), atol=1e-5)
    np.testing.assert_allclose(o_s, np.array([1.0, 1.0]), atol=1e-5)
    assert abs(_cosine(o_c, o_g)) < 0.95
    assert abs(_cosine(o_c, o_s)) < 0.95
    assert abs(_cosine(o_g, o_s)) < 0.95

    key = jax.random.PRNGKey(9)
    M33 = jax.random.normal(key, (1, 1, 3, 3))
    d_c = np.asarray(muon_c_direction(M33, polar="exact"))
    d_g = np.asarray(unfolded_muon_direction(M33, polar="exact"))
    d_s = np.asarray(spatial_polar_direction(M33))
    assert abs(_cosine(d_c, d_g)) < 0.999
    assert abs(_cosine(d_c, d_s)) < 0.999
    assert abs(_cosine(d_g, d_s)) < 0.999


def test_grouped_depthwise_unit_modulus_and_support():
    key = jax.random.PRNGKey(10)
    # Depthwise: g = Cin = Cout, stored layout (C, 1, kh, kw).
    C = 4
    M = jax.random.normal(key, (C, 1, 3, 3))
    O = muon_c_direction(M, groups=C, polar="exact")
    assert O.shape == M.shape
    G = M.reshape(C, 1, 1, 3, 3)
    Mhat = critical_fft(G)
    Ohat = critical_fft(O.reshape(C, 1, 1, 3, 3))
    # Each frequency block is 1x1, so polar is a unit-modulus phase (a sign
    # when the block is real).
    mag = np.abs(np.asarray(Ohat))
    src = np.abs(np.asarray(Mhat))
    np.testing.assert_allclose(mag, np.where(src > 1e-6, 1.0, 0.0), atol=1e-5)
    rec = critical_ifft(Ohat)
    np.testing.assert_allclose(np.asarray(rec).imag, 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(O), np.asarray(rec).real.reshape(M.shape), atol=1e-5)


def test_algorithm1_step_and_routing():
    key = jax.random.PRNGKey(11)
    kW, kG, kM = jax.random.split(key, 3)
    W = jax.random.normal(kW, (3, 2, 3, 3))
    G = jax.random.normal(kG, (3, 2, 3, 3))
    mom = jax.random.normal(kM, (3, 2, 3, 3))
    W2, mom2 = muon_c_step(W, G, mom, eta=1e-3, weight_decay=0.01, polar="exact")
    assert W2.shape == W.shape
    assert mom2.shape == mom.shape
    assert not np.allclose(np.asarray(W2), np.asarray(W))
    W3, _ = muon_c_step(W, G, mom, eta=1e-3, nesterov=False, polar="ns")
    assert W3.shape == W.shape
    assert uses_muon_c(W)
    assert not uses_muon_c(jnp.ones((3, 2, 1, 1)))
