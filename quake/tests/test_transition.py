"""Shift-register A, b, G and Lagrange-in-λ coefficients (eqs 11, 81)."""

import jax.numpy as jnp
import numpy as np

from quake.transition import lagrange_coefficients, lagrange_transition, shift_register


def test_shift_register_h2_maps_history():
    """A maps (f_{i-1}, f_{i-2}) → (a·history + b, f_{i-1}) for H=2."""
    coeffs = jnp.array([0.7, 0.3])
    bias = 0.1
    A, b, G = shift_register(coeffs, bias, window=2)
    s_prev = jnp.array([1.5, -0.4])  # (f_{i-1}, f_{i-2})
    s_pred = A @ s_prev + b
    expected_new = coeffs @ s_prev + bias
    np.testing.assert_allclose(np.asarray(s_pred[0]), np.asarray(expected_new))
    np.testing.assert_allclose(np.asarray(s_pred[1]), np.asarray(s_prev[0]))
    np.testing.assert_allclose(np.asarray(G), np.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(b), np.array([bias, 0.0]))


def test_shift_register_h3_shifts_history():
    coeffs = jnp.array([0.5, 0.3, 0.1])
    A, b, G = shift_register(coeffs, 0.0, window=3)
    s_prev = jnp.array([1.0, 2.0, 3.0])
    s_pred = A @ s_prev + b
    np.testing.assert_allclose(np.asarray(s_pred[0]), np.asarray(coeffs @ s_prev))
    np.testing.assert_allclose(np.asarray(s_pred[1:]), np.asarray(s_prev[:2]))
    np.testing.assert_allclose(np.asarray(G), np.array([1.0, 0.0, 0.0]))


def test_lagrange_linear_extrapolation():
    # λ = 3 from nodes 2, 1 → f(3) ≈ 2 f(2) − f(1)
    coeffs = lagrange_coefficients(jnp.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(np.asarray(coeffs), np.array([2.0, -1.0]), atol=1e-6)


def test_lagrange_transition_optional_x0_affine():
    lambdas = jnp.array([0.0, -1.0, -2.0])
    A, b, G = lagrange_transition(lambdas)
    np.testing.assert_allclose(np.asarray(G), np.array([1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(b[0]), 0.0)

    alpha = jnp.array([2.0, 2.0, 2.0])
    beta = jnp.array([1.0, 1.0, 1.0])
    A2, b2, _G2 = lagrange_transition(lambdas, alpha=alpha, beta=beta)
    # Constant affine x0 = 2f+1 with sum of Lagrange weights = 1 ⇒ same A, b=0.
    np.testing.assert_allclose(np.asarray(A2), np.asarray(A), atol=1e-5)
    np.testing.assert_allclose(np.asarray(b2[0]), 0.0, atol=1e-5)
