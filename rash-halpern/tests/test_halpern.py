"""Tests for Algorithm 1 and the three Appendix B models."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rash_halpern.halpern import algorithm1
from rash_halpern.models import (
    MODELS,
    model_i,
    model_i_c_derivative,
    model_i_c_objective,
    model_ii,
    model_ii_c_objective,
    model_ii_row,
    model_iii,
    worst_case_p_model_iii,
    worst_case_rho_model_i,
    worst_case_u_model_i,
    worst_case_u_model_ii,
)
from rash_halpern.operator import bias_operator, certificate_residual, gain_operator


def test_model_i_worst_case_xt():
    # Row (0, 0, 0, 1-rho, rho). Minimizing p^T v puts mass on the smaller
    # continuation, so rho = 0 when v_w < v_H and rho = 1 when v_w > v_H.
    v_w_smaller = jnp.array([0.0, 0.0, 0.0, -1.0, 4.0])
    v_h_smaller = jnp.array([0.0, 1.0, 2.0, 3.0, -5.0])
    assert worst_case_rho_model_i(v_w_smaller) == 0.0
    assert worst_case_rho_model_i(v_h_smaller) == 1.0

    def continuation(rho, v):
        return (1.0 - rho) * float(v[3]) + rho * float(v[4])

    for v in (v_w_smaller, v_h_smaller):
        rho = worst_case_rho_model_i(v)
        assert continuation(rho, v) <= continuation(0.0, v) + 1e-15
        assert continuation(rho, v) <= continuation(1.0, v) + 1e-15


def test_model_i_worst_case_xc():
    # Interior minimum: derivative changes from negative to positive in (0, 1/2).
    v_interior = jnp.array([0.0, -0.3, 1.0, 0.0, 0.0])
    d0 = model_i_c_derivative(0.0, v_interior)
    d_hi = model_i_c_derivative(0.5, v_interior)
    assert d0 < 0.0 < d_hi
    u_root = math.sqrt(0.1)
    assert 0.0 < u_root < 0.5
    u = worst_case_u_model_i(v_interior)
    assert abs(u - u_root) < 1e-8
    phi_u = model_i_c_objective(u, v_interior)
    assert phi_u < model_i_c_objective(0.0, v_interior) - 1e-8
    assert phi_u < model_i_c_objective(0.5, v_interior) - 1e-8

    # Derivative does not change sign: an endpoint wins.
    v_increasing = jnp.array([0.0, 1.0, 2.0, 0.0, 0.0])
    assert model_i_c_derivative(0.0, v_increasing) > 0.0
    assert model_i_c_derivative(0.5, v_increasing) > 0.0
    u_left = worst_case_u_model_i(v_increasing)
    assert abs(u_left - 0.0) < 1e-12
    assert model_i_c_objective(u_left, v_increasing) <= model_i_c_objective(0.5, v_increasing)

    v_decreasing = jnp.array([0.0, -1.0, -1.0, 0.0, 0.0])
    assert model_i_c_derivative(0.0, v_decreasing) < 0.0
    assert model_i_c_derivative(0.5, v_decreasing) < 0.0
    u_right = worst_case_u_model_i(v_decreasing)
    assert abs(u_right - 0.5) < 1e-12
    assert model_i_c_objective(u_right, v_decreasing) <= model_i_c_objective(0.0, v_decreasing)

    # A sign-changing root that is a maximum must not be returned.
    v_max = jnp.array([0.0, 0.3, -1.0, 0.0, 0.0])
    u_max_root = math.sqrt(0.1)
    assert 0.0 < u_max_root < 0.5
    assert model_i_c_derivative(0.0, v_max) > 0.0
    assert model_i_c_derivative(0.5, v_max) < 0.0
    u_end = worst_case_u_model_i(v_max)
    assert abs(u_end - 0.0) < 1e-12
    assert model_i_c_objective(u_end, v_max) < model_i_c_objective(u_max_root, v_max)


def _dense_grid_min(phi, lo: float = 0.0, hi: float = 1.0, n: int = 2001) -> float:
    """Argmin of phi on a dense grid, polished by a local parabola.

    An exact quadratic is determined by three samples, so the vertex matches
    the continuous minimizer far inside 1e-8 when the bowl is visible.
    """
    grid = np.linspace(lo, hi, n)
    vals = np.array([phi(float(u)) for u in grid], dtype=np.float64)
    index = int(np.argmin(vals))
    if index == 0 or index == n - 1:
        return float(grid[index])
    y0, y1, y2 = vals[index - 1], vals[index], vals[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(grid[index])
    step = float(grid[index] - grid[index - 1])
    offset = 0.5 * step * (y0 - y2) / denom
    return float(grid[index] + offset)


def test_model_ii_row_and_worst_u():
    grid = np.linspace(0.0, 1.0, 21)
    for u in grid:
        row = model_ii_row(float(u))
        assert abs(float(jnp.sum(row)) - 1.0) < 1e-12
        assert float(jnp.min(row)) >= -1e-12

    vectors = [
        jnp.array([0.0, 0.0, 0.7, 0.0, 0.0, 1.0]),
        jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        jnp.array([0.0, 0.0, 0.2, 0.0, 1.0, 0.0]),
        jnp.array([1.0, -2.0, 0.3, 4.0, -1.0, 2.0]),
        jnp.array([0.2, -0.4, 1.5, -0.2, 0.3, -0.7]),
        jnp.zeros(6),
    ]
    for v in vectors:
        u = worst_case_u_model_ii(v)
        assert 0.0 <= u <= 1.0

        def phi(t, v=v):
            return model_ii_c_objective(t, v)

        u_grid = _dense_grid_min(phi, 0.0, 1.0)
        assert abs(u - u_grid) < 1e-8
        assert abs(phi(u) - phi(u_grid)) < 1e-8


def test_model_iii_worst_case_p():
    v_h_larger = jnp.array([0.0, 0.0, 1.0])
    v_l_larger = jnp.array([0.0, 2.0, -1.0])
    assert worst_case_p_model_iii(v_h_larger) == 0.1
    assert worst_case_p_model_iii(v_l_larger) == 0.9

    def continuation(p, v):
        return (1.0 - p) * float(v[1]) + p * float(v[2])

    for v, p_expected in ((v_h_larger, 0.1), (v_l_larger, 0.9)):
        p = worst_case_p_model_iii(v)
        assert p == p_expected
        for other in (0.1, 0.5, 0.9):
            assert continuation(p, v) <= continuation(other, v) + 1e-15


def test_certificate():
    # Equation 9: g = T-bar g and g + h = L_g h. The unrestricted operator
    # does not satisfy T h = h + g* on Models I and II, because inactive
    # actions and off-face rows can win a one-step comparison.
    for model in MODELS:
        gain_err = jnp.max(jnp.abs(gain_operator(model, model.g_star) - model.g_star))
        bias_err = jnp.max(jnp.abs(certificate_residual(model)))
        lg = bias_operator(model, model.g_star, model.h)
        assert float(gain_err) < 1e-8
        assert float(bias_err) < 1e-8
        assert float(jnp.max(jnp.abs(lg - model.h - model.g_star))) < 1e-8


def test_model_i_action_switch():
    # Paper switch for the printed update: t at N=1, 2 and c from N=3.
    expected = {1: "t", 2: "t", 3: "c", 4: "c", 32: "c"}
    for n, name in expected.items():
        result = algorithm1(model_i, n, nominal=False)
        assert model_i.action_names[int(result.pi[0])] == name

    # Lock the printed Phase II weights on the N=1 closed form.
    result = algorithm1(model_i, 1, nominal=False)
    assert jnp.allclose(result.g_hat, jnp.array([10.0, -1.0, 1.0, -2.0, 5.0]))
    assert jnp.allclose(result.Z, jnp.array([6.0, 8.0 / 3.0, 1.0, -2.0, 5.0]))


def test_models_ii_iii_optimal_actions():
    for n in (1, 2, 4, 16):
        result_ii = algorithm1(model_ii, n, nominal=False)
        result_iii = algorithm1(model_iii, n, nominal=False)
        assert model_ii.action_names[int(result_ii.pi[0])] == "c"
        assert model_iii.action_names[int(result_iii.pi[0])] == "safe"
        assert int(result_ii.pi[0]) == model_ii.optimal_action_at_x
        assert int(result_iii.pi[0]) == model_iii.optimal_action_at_x


def test_gain_error_decreases():
    for model in MODELS:
        err4 = float(jnp.max(jnp.abs(algorithm1(model, 4).g_hat - model.g_star)))
        err64 = float(jnp.max(jnp.abs(algorithm1(model, 64).g_hat - model.g_star)))
        assert err64 < err4
        assert err64 < 0.05


def test_nominal_ablation_selects_risky():
    # The nominal mistake is already present at moderate budgets.
    for n in (16, 64):
        nominal_i = algorithm1(model_i, n, nominal=True)
        nominal_iii = algorithm1(model_iii, n, nominal=True)
        assert model_i.action_names[int(nominal_i.pi[0])] == "t"
        assert model_iii.action_names[int(nominal_iii.pi[0])] == "risky"
