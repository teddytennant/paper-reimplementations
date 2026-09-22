"""Fixed numeric checks for the proximal residual allocator."""

import jax
import jax.numpy as jnp
import numpy as np

from proxvf.alloc import (
    allocate,
    entropy,
    hinge_residual,
    quadratic_H_inv_ones,
    quadratic_allocate,
    quadratic_path,
)


def _as_vec(z):
    return np.asarray(z, dtype=float).reshape(-1)


def test_quadratic_H_inv_ones_fixed():
    h_inv_ones = quadratic_H_inv_ones()
    got = _as_vec(h_inv_ones)
    assert got.shape == (2,)
    assert np.allclose(got, (2.0, -1.0), atol=1e-6)


def test_quadratic_path_closed_form_on_unit_interval():
    # For this H and a = (0, 1), the simplex minimizer is (B, 0).
    cases = (
        (0.0, (0.0, 0.0)),
        (0.25, (0.25, 0.0)),
        (1.0, (1.0, 0.0)),
        (1.25, (1.25, 0.0)),
        (1.5, (1.5, 0.0)),
        (2.0, (2.0, 0.0)),
    )
    for budget, expected in cases:
        got = _as_vec(quadratic_path(budget))
        assert got.shape == (2,)
        assert np.allclose(got, expected, atol=1e-5)


def test_quadratic_allocate_matches_path_budget_and_sign():
    # 2-d quadratic H = [[2, 3], [3, 5]], a = (0, 1). I only sets shape.
    eye = jnp.array([0.0, 1.0])
    for budget in (0.0, 0.4, 1.0, 1.7, 2.0):
        z = quadratic_allocate(eye, budget)
        got = _as_vec(z)
        assert got.shape == (2,)
        assert np.all(got >= -1e-5)
        assert np.allclose(got.sum(), budget, atol=1e-4)
        assert np.allclose(got, _as_vec(quadratic_path(budget)), atol=1e-4)


def test_entropy_fixed_xlogx():
    x = jnp.array([0.5, 1.0, 2.0])
    got = np.asarray(entropy(x), dtype=float)
    element = np.asarray(x * jnp.log(x), dtype=float)
    total = float(np.sum(element))
    if got.shape == ():
        assert np.allclose(got, total, atol=1e-5)
    else:
        assert got.shape == element.shape
        assert np.allclose(got, element, atol=1e-5)


def test_hinge_residual_softplus_eta_fixed():
    x = jnp.array([0.25, 0.75, 1.5])
    knots = jnp.array([0.0, 0.5, 1.0])
    eta = jnp.array(0.0)
    residual = np.asarray(hinge_residual(x, eta, knots), dtype=float)
    assert residual.shape == () or residual.shape == x.shape or residual.shape == (x.shape[0], knots.shape[0])
    assert np.isfinite(residual).all()

    # eta is passed through softplus, so a large negative weight stays finite
    # and is strictly smaller in scale than a large positive weight.
    low = np.asarray(hinge_residual(x, jnp.array(-4.0), knots), dtype=float)
    high = np.asarray(hinge_residual(x, jnp.array(4.0), knots), dtype=float)
    assert np.isfinite(low).all()
    assert np.isfinite(high).all()
    # softplus(-4) ≈ 0.018, softplus(4) ≈ 4.018; residual scales with that factor
    # when the hinge feature of x is nonzero. Compare sums of absolute values.
    assert np.sum(np.abs(high)) > np.sum(np.abs(low))


def _hinge_slope(z, eta, knots):
    def scalar_residual(xx):
        return jnp.sum(hinge_residual(xx, eta, knots))

    return jax.grad(scalar_residual)(z)


def _marginal(x, c, tau, eta, knots):
    return c + tau * (jnp.log(x) + 1.0) + _hinge_slope(x, eta, knots)


def test_allocate_budget_nonnegative_shape_and_support_stationarity():
    # Fixed interior example. Positive tau keeps the entropy barrier active.
    investments = jnp.array([0.2, 0.5, 0.9])
    costs = jnp.array([0.15, -0.05, 0.40])
    budget = 1.25
    tau = 0.2
    eta = jnp.array(0.3)
    knots = jnp.array([0.0, 0.4, 1.0])

    z = allocate(investments, costs, budget, tau, eta, knots, steps=48)
    got = np.asarray(z, dtype=float)
    assert got.shape == np.asarray(investments).shape
    assert np.all(got >= -1e-6)
    assert np.allclose(got.sum(), budget, atol=1e-3)

    support = got > 1e-3
    assert np.any(support)
    levels = investments + z
    marginals = np.asarray(
        _marginal(levels, costs, tau, eta, knots), dtype=float
    ).reshape(-1)
    active = marginals[support.reshape(-1)]
    assert np.all(np.isfinite(active))
    # Dual bisection: equalized marginals on the support. Do not loosen.
    assert float(np.max(active) - np.min(active)) <= 5e-2


def test_allocate_eta_gradient_finite():
    investments = jnp.array([1.0, 0.0, 0.5])
    costs = jnp.array([0.1, 0.2, 0.05])
    budget = 1.0
    tau = 0.15
    knots = jnp.array([0.0, 0.5])

    def loss(eta):
        z = allocate(investments, costs, budget, tau, eta, knots, steps=48)
        return jnp.sum(z) * eta

    grad_eta = jax.grad(loss)(jnp.array(0.25))
    assert np.isfinite(float(np.asarray(grad_eta)))
