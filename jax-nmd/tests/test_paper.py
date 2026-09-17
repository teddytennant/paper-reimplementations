"""Equation checks against numpy for arXiv:2609.14402."""

from __future__ import annotations

import numpy as np
import pytest

from jax_nmd.loss import ell_gauge, gauge_loss, log_z_loss, scattering_mae, total_loss
from jax_nmd.poles import poles_from_raw
from jax_nmd.synthesize import dominance_score, synthesize_z, z_to_s


def np_softplus(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def np_synthesize_z(omega, poles, left, right):
    omega = np.asarray(omega, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.complex128)
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    f, k, n = omega.shape[0], left.shape[0], left.shape[1]
    z = np.zeros((f, n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            acc = np.zeros(f, dtype=np.complex128)
            for kk in range(k):
                acc = acc + left[kk, i] * right[kk, j] / (1j * omega - poles[kk])
            z[:, i, j] = acc
    return z


def np_z_to_s(z, z0=50.0):
    z = np.asarray(z, dtype=np.complex128)
    n = z.shape[-1]
    eye = np.eye(n, dtype=np.complex128)
    out = np.empty_like(z)
    it = np.ndindex(z.shape[:-2])
    for idx in it:
        zz = z[idx]
        out[idx] = (zz - z0 * eye) @ np.linalg.inv(zz + z0 * eye)
    return out


def np_dominance(omega, poles, left, right):
    omega = np.asarray(omega, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.complex128)
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    k = poles.shape[0]
    scores = np.zeros(k, dtype=np.float64)
    for kk in range(k):
        mk = np.linalg.norm(left[kk]) * np.linalg.norm(right[kk])
        gain = np.sqrt(np.sum(1.0 / np.abs(1j * omega - poles[kk]) ** 2))
        scores[kk] = mk * gain
    return scores


def _random_system(rng, f=13, k=4, n=3):
    omega = rng.uniform(1.0, 10.0, size=f)
    u = rng.normal(size=k)
    v = rng.normal(size=k) * 3.0
    poles = -np_softplus(u) + 1j * v
    mag_l = np_softplus(rng.normal(size=(k, n)))
    mag_r = np_softplus(rng.normal(size=(k, n)))
    ph_l = rng.uniform(-np.pi, np.pi, size=(k, n))
    ph_r = rng.uniform(-np.pi, np.pi, size=(k, n))
    left = mag_l * np.exp(1j * ph_l)
    right = mag_r * np.exp(1j * ph_r)
    return omega, poles, left, right, u, v


def test_synthesize_z_matches_numpy_eq4():
    rng = np.random.default_rng(0)
    omega, poles, left, right, _, _ = _random_system(rng)
    z_np = np_synthesize_z(omega, poles, left, right)
    z_jax = np.asarray(synthesize_z(omega, poles, left, right))
    np.testing.assert_allclose(z_jax, z_np, atol=1e-5, rtol=1e-5)


def test_poles_from_raw_eq3():
    rng = np.random.default_rng(1)
    u = rng.normal(size=8)
    v = rng.normal(size=8) * 2.0
    p = np.asarray(poles_from_raw(u, v))
    gamma = np_softplus(u)
    assert np.all(gamma > 0.0)
    assert np.all(np.real(p) < 0.0)
    np.testing.assert_allclose(np.real(p), -gamma, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.imag(p), v, atol=1e-5, rtol=1e-5)
    p_direct = -gamma + 1j * v
    np.testing.assert_allclose(p, p_direct, atol=1e-5, rtol=1e-5)


def test_z_to_s_matches_numpy_bilinear():
    rng = np.random.default_rng(2)
    omega, poles, left, right, _, _ = _random_system(rng, n=2)
    z = np_synthesize_z(omega, poles, left, right)
    s_np = np_z_to_s(z, z0=50.0)
    s_jax = np.asarray(z_to_s(z, z0=50.0))
    np.testing.assert_allclose(s_jax, s_np, atol=1e-5, rtol=1e-5)


def test_gauge_ell_eq5():
    eps = 1e-6
    pos = np.asarray(ell_gauge(3.0 + 0.0j, eps=eps))
    neg = np.asarray(ell_gauge(-3.0 + 0.0j, eps=eps))
    imag = np.asarray(ell_gauge(3.0j, eps=eps))
    np.testing.assert_allclose(pos, 1.0 - 3.0 / (3.0 + eps), atol=1e-6)
    np.testing.assert_allclose(neg, 1.0 - (-3.0) / (3.0 + eps), atol=1e-6)
    np.testing.assert_allclose(imag, 1.0, atol=1e-6)
    assert pos < 1e-5
    assert neg > 0.0
    assert imag > 0.0


def test_total_loss_eq6_composition():
    rng = np.random.default_rng(3)
    omega, poles, left, right, _, _ = _random_system(rng, n=2)
    z_gt = np_synthesize_z(omega, poles, left, right)
    s_gt = np_z_to_s(z_gt)
    z_pr = z_gt * 1.1 + 0.01
    s_pr = np_z_to_s(z_pr)
    is_anchor = np.array([True, False])
    loss, parts = total_loss(s_pr, s_gt, z_pr, z_gt, left, right, is_anchor)
    ls = np.asarray(scattering_mae(s_pr, s_gt))
    lz = np.asarray(log_z_loss(z_pr, z_gt, eps=1e-2))
    lg = np.asarray(gauge_loss(left, right, is_anchor))
    composed = ls + 0.1 * lz + 0.5 * lg
    np.testing.assert_allclose(np.asarray(loss), composed, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(parts["L_S"]), ls, atol=1e-6)
    np.testing.assert_allclose(np.asarray(parts["L_Z_log"]), lz, atol=1e-6)
    np.testing.assert_allclose(np.asarray(parts["L_gauge"]), lg, atol=1e-6)


def test_log_z_loss_formula():
    rng = np.random.default_rng(4)
    z_gt = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
    z_pr = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
    eps = 1e-2
    expected = np.mean(np.abs(np.log(np.abs(z_pr) + eps) - np.log(np.abs(z_gt) + eps)))
    got = np.asarray(log_z_loss(z_pr, z_gt, eps=eps))
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_dominance_score_eq18_matches_numpy():
    rng = np.random.default_rng(5)
    omega, poles, left, right, _, _ = _random_system(rng, n=2)
    d_np = np_dominance(omega, poles, left, right)
    d_jax = np.asarray(dominance_score(omega, poles, left, right))
    np.testing.assert_allclose(d_jax, d_np, atol=1e-5, rtol=1e-5)


def test_dominance_ranks_inband_high_residue():
    omega = np.linspace(1.0, 10.0, 36)
    poles = np.array([-0.2 + 5.0j, -0.2 + 80.0j], dtype=np.complex128)
    left = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.complex128)
    right = left.copy()
    scores = np.asarray(dominance_score(omega, poles, left, right))
    assert scores[0] > scores[1]
