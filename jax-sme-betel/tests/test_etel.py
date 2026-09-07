"""Checks on the exponentially tilted empirical likelihood solve."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sme_betel.etel import etel_weights, log_etel, solve_tilt


def _moment_matrix(key, n=300, p=3, shift=0.35):
    g = jax.random.normal(key, (n, p))
    g = g.at[:, 1].mul(2.5)
    return g + shift


def test_weights_are_a_probability_vector_meeting_the_constraint():
    """Equation (2.6) asks for weights on the simplex with sum_i w_i g_i = 0."""
    g = _moment_matrix(jax.random.key(0))
    w = etel_weights(g)
    assert float(jnp.min(w)) > 0.0
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-13
    resid = w @ g
    scale = float(jnp.max(jnp.abs(g)))
    assert float(jnp.max(jnp.abs(resid))) < 1e-11 * scale


def test_tilt_is_exactly_zero_when_the_sample_moments_already_balance():
    """Uniform weights already solve eq (2.6) when the g_i average to zero."""
    g = _moment_matrix(jax.random.key(1), shift=0.0)
    g = g - g.mean(axis=0)
    sol = solve_tilt(g)
    n = g.shape[0]
    assert float(jnp.max(jnp.abs(sol.lam))) == 0.0
    w = jnp.exp(sol.log_w)
    assert float(jnp.max(jnp.abs(w - 1.0 / n))) < 1e-15
    assert abs(float(sol.log_lik) + n * np.log(n)) < 1e-9
    assert bool(sol.feasible)


def test_origin_outside_the_convex_hull_is_reported_infeasible():
    """The eq (2.5) indicator, and the discard branch of Algorithm S1."""
    key = jax.random.key(2)
    g = jnp.abs(jax.random.normal(key, (200, 2))) + 0.1
    sol = solve_tilt(g)
    assert not bool(sol.feasible)
    assert float(log_etel(g)) == -np.inf

    # In one dimension the origin is interior to the hull exactly when the
    # sample straddles zero, so a shift can be walked across the boundary.
    v = jax.random.normal(jax.random.key(3), (200, 1))
    lo, hi = float(jnp.min(v)), float(jnp.max(v))
    assert bool(solve_tilt(v - 0.5 * (lo + hi)).feasible)
    assert not bool(solve_tilt(v - lo + 1e-3).feasible)
    assert not bool(solve_tilt(v - hi - 1e-3).feasible)


def test_uniform_weights_maximize_the_log_etel_function():
    """sum_i log w_i <= -n log n by AM-GM, with equality only at uniform w."""
    n = 250
    balanced = jax.random.normal(jax.random.key(4), (n, 2))
    balanced = balanced - balanced.mean(axis=0)
    top = -n * np.log(n)
    assert abs(float(log_etel(balanced)) - top) < 1e-9
    for shift in (0.05, 0.2, 0.5):
        assert float(log_etel(balanced + shift)) < top - 1e-6


def test_dual_solution_agrees_with_a_generic_optimizer():
    """Cross-check lambda_hat(theta) against scipy on the eq (2.6) objective."""
    g = np.asarray(_moment_matrix(jax.random.key(5), n=150, p=2, shift=0.25))
    n = g.shape[0]

    def obj(lam):
        return float(np.mean(np.exp(g @ lam)))

    def jac(lam):
        e = np.exp(g @ lam)
        return g.T @ e / n

    res = minimize(obj, np.zeros(g.shape[1]), jac=jac, method="BFGS", tol=1e-14)
    ours = np.asarray(solve_tilt(jnp.asarray(g)).lam)
    assert np.max(np.abs(res.x - ours)) < 1e-6


def test_weights_are_invariant_to_rescaling_the_moment_columns():
    """The internal column scaling must not change omega(theta)."""
    g = _moment_matrix(jax.random.key(6), n=200, p=3, shift=0.3)
    scale = jnp.array([1.0, 1e3, 1e-3])
    a = etel_weights(g)
    b = etel_weights(g * scale)
    assert float(jnp.max(jnp.abs(a - b))) < 1e-11
