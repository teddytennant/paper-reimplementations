"""MSFC / SAR / probe loss unit tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_gralod.losses import (
    calibration_ramp,
    l_cal,
    l_probe,
    l_str,
    msfc_costs,
    msfc_distribution,
    msfc_target,
    stage_structure_penalty,
    total_loss,
)


def test_calibration_ramp_default_is_one():
    w = calibration_ramp(0, warmup_steps=0, lambda_cal_max=0.10)
    np.testing.assert_allclose(float(w), 0.10)
    w2 = calibration_ramp(5, warmup_steps=10, lambda_cal_max=0.10)
    np.testing.assert_allclose(float(w2), 0.05)
    w3 = calibration_ramp(20, warmup_steps=10, lambda_cal_max=0.10)
    np.testing.assert_allclose(float(w3), 0.10)


def test_msfc_softmax_and_target():
    errors = jnp.array([0.2, 0.1, 0.4], dtype=jnp.float32)
    lam_star, q, c = msfc_target(errors[None, None, None, :], gamma=0.05, tau=0.20)
    q = q[0, 0, 0]
    c = c[0, 0, 0]
    np.testing.assert_allclose(q, jax.nn.softmax(-c / 0.20), atol=1e-6)
    levels = jnp.arange(3.0)
    np.testing.assert_allclose(float(lam_star[0, 0, 0]), float(jnp.sum(q * levels)), atol=1e-6)


def test_l_str_zero_when_lambda_constant():
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 4, 3))
    lam = jnp.ones((1, 4, 4))
    pen = stage_structure_penalty(lam, x, beta=10.0)
    np.testing.assert_allclose(float(pen), 0.0, atol=1e-6)


def test_total_loss_weights():
    rec, cal, st, pr = 1.0, 2.0, 3.0, 4.0
    loss = total_loss(rec, cal, st, pr, lambda_cal=0.10, lambda_str=0.05, lambda_probe=0.10)
    np.testing.assert_allclose(float(loss), 1.0 + 0.10 * 2.0 + 0.05 * 3.0 + 0.10 * 4.0)
