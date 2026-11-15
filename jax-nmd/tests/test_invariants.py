"""Algebraic priors: sign gauge, submatrix consistency, reciprocity, anchors."""

from __future__ import annotations

import numpy as np

from jax_nmd.loss import ell_gauge, gauge_loss
from jax_nmd.synthesize import synthesize_z, z_to_s


def test_sign_invariance_neg_l_neg_r():
    rng = np.random.default_rng(10)
    f, k, n = 9, 3, 3
    omega = rng.uniform(1.0, 8.0, size=f)
    poles = -np.abs(rng.normal(size=k)) - 0.1 + 1j * rng.normal(size=k)
    left = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    right = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    z1 = np.asarray(synthesize_z(omega, poles, left, right))
    z2 = np.asarray(synthesize_z(omega, poles, -left, -right))
    np.testing.assert_allclose(z1, z2, atol=1e-5, rtol=1e-5)


def test_submatrix_consistency():
    rng = np.random.default_rng(11)
    f, k, n = 7, 4, 4
    omega = rng.uniform(1.0, 8.0, size=f)
    poles = -np.abs(rng.normal(size=k)) - 0.2 + 1j * rng.normal(size=k)
    left = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    right = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    z_full = np.asarray(synthesize_z(omega, poles, left, right))
    subset = np.array([0, 2, 3])
    z_sub = np.asarray(
        synthesize_z(omega, poles, left[:, subset], right[:, subset])
    )
    z_slice = z_full[:, subset][:, :, subset]
    np.testing.assert_allclose(z_sub, z_slice, atol=1e-5, rtol=1e-5)


def test_reciprocal_z_gives_reciprocal_s():
    rng = np.random.default_rng(12)
    f, k, n = 8, 3, 2
    omega = rng.uniform(1.0, 8.0, size=f)
    poles = -np.abs(rng.normal(size=k)) - 0.15 + 1j * rng.normal(size=k)
    coup = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    z = np.asarray(synthesize_z(omega, poles, coup, coup))
    np.testing.assert_allclose(z, np.swapaxes(z, -1, -2), atol=1e-5, rtol=1e-5)
    s = np.asarray(z_to_s(z, z0=50.0))
    np.testing.assert_allclose(s, np.swapaxes(s, -1, -2), atol=1e-5, rtol=1e-5)


def test_gauge_positive_real_near_zero():
    a = 2.5 + 0.0j
    val = float(np.asarray(ell_gauge(a)))
    assert val < 1e-5


def test_gauge_negative_real_and_imaginary_positive():
    assert float(np.asarray(ell_gauge(-1.5 + 0.0j))) > 0.0
    assert float(np.asarray(ell_gauge(1.5j))) > 0.0


def test_gauge_loss_zero_when_no_anchor_ports():
    rng = np.random.default_rng(13)
    k, n = 4, 3
    left = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    right = rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n))
    is_anchor = np.zeros((n,), dtype=bool)
    lg = float(np.asarray(gauge_loss(left, right, is_anchor)))
    assert lg == 0.0


def test_gauge_loss_positive_on_anchor_with_negative_real():
    k, n = 2, 2
    left = np.full((k, n), -1.0 + 0.0j)
    right = np.full((k, n), -1.0 + 0.0j)
    is_anchor = np.array([True, False])
    lg = float(np.asarray(gauge_loss(left, right, is_anchor)))
    assert lg > 0.0


def test_gauge_loss_near_zero_for_positive_real_anchor():
    k, n = 2, 2
    left = np.full((k, n), 1.7 + 0.0j)
    right = np.full((k, n), 0.9 + 0.0j)
    is_anchor = np.array([True, True])
    lg = float(np.asarray(gauge_loss(left, right, is_anchor)))
    assert lg < 1e-5
