"""Properties of the smoothed-entropy mirror step and the step-size schedule."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from layered_omd.mirror import (
    bregman_divergence,
    bregman_diameter,
    prox_constant,
    prox_step,
    step_sizes,
)


def _objective(x, grad, base, eta, lam):
    x = np.asarray(x, dtype=float)
    return float(eta * grad @ x + np.sum((x + lam) * np.log((x + lam) / (base + lam))))


def _random_instance(rng):
    n = int(rng.integers(2, 6))
    base = rng.dirichlet(np.ones(n))
    grad = rng.normal(scale=3.0, size=n)
    eta = float(rng.uniform(0.01, 5.0))
    lam = float(rng.choice([1.0 / n, 0.05, 0.5]))
    zeta = float(rng.choice([0.0, 0.05, 0.3]))
    return n, base, grad, eta, lam, zeta / n


def test_prox_matches_a_general_purpose_convex_solver():
    """The water-filling formula solves the same program SLSQP does."""
    rng = np.random.default_rng(0)
    converged = 0
    for _ in range(200):
        n, base, grad, eta, lam, floor = _random_instance(rng)
        ours = np.asarray(prox_step(jnp.asarray(grad), jnp.asarray(base), eta, lam, floor))
        res = minimize(
            _objective,
            np.full(n, 1.0 / n),
            args=(grad, base, eta, lam),
            method="SLSQP",
            bounds=[(floor, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        # Our point never loses to SLSQP's, whether or not SLSQP declared success.
        assert _objective(ours, grad, base, eta, lam) <= res.fun + 1e-9
        if res.success:
            converged += 1
            # SLSQP is the loose one here, so compare at its tolerance.
            assert np.max(np.abs(ours - res.x)) < 1e-4
    assert converged > 150


def test_prox_reduces_to_exponential_weights_without_smoothing():
    """lambda = 0 and no floor: the step is exactly x = y exp(-eta g) / Z."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(2, 7))
        base = rng.dirichlet(np.ones(n))
        grad = rng.normal(scale=2.0, size=n)
        eta = float(rng.uniform(0.05, 3.0))
        weights = base * np.exp(-eta * grad)
        expected = weights / weights.sum()
        ours = np.asarray(prox_step(jnp.asarray(grad), jnp.asarray(base), eta, 0.0, 0.0))
        assert np.max(np.abs(ours - expected)) < 1e-13


def test_prox_matches_the_interior_closed_form_when_nothing_clips():
    """With a small step every coordinate stays interior: x = rho w / sum(w) - lambda."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        base = rng.dirichlet(np.ones(n)) * 0.5 + 0.5 / n
        grad = rng.normal(size=n)
        eta, lam = 1e-3, 1.0 / n
        weights = (base + lam) * np.exp(-eta * grad)
        expected = prox_constant(n, lam) * weights / weights.sum() - lam
        assert expected.min() > 0.0
        ours = np.asarray(prox_step(jnp.asarray(grad), jnp.asarray(base), eta, lam, 0.0))
        assert np.max(np.abs(ours - expected)) < 1e-13


def test_prox_output_stays_on_the_simplex_and_above_the_floor():
    rng = np.random.default_rng(3)
    for _ in range(400):
        n, base, grad, eta, lam, floor = _random_instance(rng)
        # very large steps push most of the mass onto one coordinate
        eta *= float(rng.choice([1.0, 50.0]))
        ours = np.asarray(prox_step(jnp.asarray(grad), jnp.asarray(base), eta, lam, floor))
        assert abs(ours.sum() - 1.0) < 1e-12
        assert ours.min() >= floor - 1e-15


def test_prox_is_lipschitz_in_the_gradient():
    """Lemma 5: ||P_y(g) - P_y(g')||_1 <= eta rho_i ||g - g'||_inf."""
    rng = np.random.default_rng(4)
    for _ in range(300):
        n, base, grad, eta, lam, floor = _random_instance(rng)
        other = grad + rng.normal(scale=1.0, size=n)
        a = np.asarray(prox_step(jnp.asarray(grad), jnp.asarray(base), eta, lam, floor))
        b = np.asarray(prox_step(jnp.asarray(other), jnp.asarray(base), eta, lam, floor))
        bound = eta * prox_constant(n, lam) * np.max(np.abs(grad - other))
        assert np.sum(np.abs(a - b)) <= bound + 1e-12


def test_prox_with_zero_gradient_is_the_identity():
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        base = rng.dirichlet(np.ones(n))
        ours = np.asarray(prox_step(jnp.zeros(n), jnp.asarray(base), 0.7, 1.0 / n, 0.0))
        assert np.max(np.abs(ours - base)) < 1e-13


def test_bregman_divergence_is_nonnegative_and_bounded_by_the_diameter():
    """Lemma 4: 0 <= D_i(x, y) <= B_i = rho_i log((1 + lambda)/lambda)."""
    rng = np.random.default_rng(6)
    for _ in range(300):
        n = int(rng.integers(2, 7))
        lam = float(rng.choice([1.0 / n, 0.05]))
        x = rng.dirichlet(np.ones(n) * 0.3)
        y = rng.dirichlet(np.ones(n) * 0.3)
        div = float(bregman_divergence(jnp.asarray(x), jnp.asarray(y), lam))
        assert -1e-13 <= div <= float(bregman_diameter(n, lam)) + 1e-12
        # strong convexity, also Lemma 4
        assert div >= np.sum(np.abs(x - y)) ** 2 / (2.0 * prox_constant(n, lam)) - 1e-12


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 12])
def test_step_size_schedule_increases_with_the_layer(horizon):
    """alpha_h = (3(H-h)+1)/(3H+1) decreases, so eta_h = eta_0 T^{-alpha_h} increases."""
    eta = np.asarray(step_sizes(horizon, 1000, 0.5))
    assert eta.shape == (horizon,)
    assert np.all(np.diff(eta) > 0) or horizon == 1
    assert eta[-1] == pytest.approx(0.5 * 1000.0 ** (-1.0 / (3 * horizon + 1)))
    assert eta[0] == pytest.approx(
        0.5 * 1000.0 ** (-(3.0 * horizon - 2.0) / (3 * horizon + 1))
    )
    assert np.all(eta <= 0.5)
