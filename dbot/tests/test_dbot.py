"""Checks for the enumerated D-BOT kernel. CPU only, n <= 4."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dbot import (
    bisimulation_lp,
    envelope_grad,
    fit_kernel,
    occupancy,
    softmax_kernel,
)


def _indicator_cost(n):
    return 1.0 - jnp.eye(n)


def test_softmax_and_occupancy():
    logits = jnp.array([[0.0, 1.0, -1.0], [2.0, 2.0, 2.0], [-3.0, 0.5, 0.5]])
    kernel = softmax_kernel(logits)
    np.testing.assert_allclose(np.sum(kernel, axis=1), 1.0, atol=1e-12)
    assert np.all(np.asarray(kernel) > 0.0)

    # Hand solution of (I - gamma P^T) rho = (1 - gamma) nu0.
    # P = [[0.8, 0.2], [0.3, 0.7]], gamma = 0.5, nu0 = [1, 0]
    # [[0.6, -0.15], [-0.1, 0.65]] rho = [0.5, 0]
    # rho = [13/15, 2/15]
    # nu(y, y') = rho(y) * P(y'|y)
    known = jnp.array([[0.8, 0.2], [0.3, 0.7]])
    nu0 = jnp.array([1.0, 0.0])
    nu = occupancy(known, nu0, 0.5)
    expected = np.array(
        [
            [52.0 / 75.0, 13.0 / 75.0],
            [1.0 / 25.0, 7.0 / 75.0],
        ]
    )
    np.testing.assert_allclose(nu, expected, atol=1e-12)
    np.testing.assert_allclose(np.sum(nu), 1.0, atol=1e-12)


def test_identical_kernels_have_zero_distance():
    kernel = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
        ]
    )
    gamma = 0.9
    nu0 = jnp.array([1.0, 0.0, 0.0])
    nu = occupancy(kernel, nu0, gamma)
    result = bisimulation_lp(nu, nu, _indicator_cost(3), gamma, 0, 0)
    assert result["success"]
    assert result["distance"] < 1e-5


def test_self_loop_versus_moving_chain():
    gamma = 0.9
    nu0 = jnp.array([1.0, 0.0, 0.0])
    stay = jnp.eye(3)
    move = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    nu_stay = occupancy(stay, nu0, gamma)
    nu_move = occupancy(move, nu0, gamma)
    result = bisimulation_lp(nu_stay, nu_move, _indicator_cost(3), gamma, 0, 0)
    assert result["success"]
    assert result["distance"] > 0.05


def test_envelope_gradient_matches_finite_difference():
    gamma = 0.8
    n = 3
    target = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.25, 0.25, 0.5],
        ]
    )
    nu0 = jnp.array([1.0, 0.0, 0.0])
    nu_x = occupancy(target, nu0, gamma)
    cost = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    logits = jax.random.normal(jax.random.PRNGKey(0), (n, n))

    def distance_of(weights):
        nu_y = occupancy(softmax_kernel(weights), nu0, gamma)
        result = bisimulation_lp(nu_x, nu_y, cost, gamma, 0, 0)
        assert result["success"]
        return result["distance"]

    center = bisimulation_lp(
        nu_x,
        occupancy(softmax_kernel(logits), nu0, gamma),
        cost,
        gamma,
        0,
        0,
    )
    assert center["success"]
    analytic = np.asarray(
        envelope_grad(logits, center["lam_y"], center["alpha_y"], nu0, gamma)
    )

    step = 1e-4
    finite = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bump = jnp.zeros((n, n)).at[i, j].set(step)
            finite[i, j] = (distance_of(logits + bump) - distance_of(logits - bump)) / (
                2.0 * step
            )

    assert np.max(np.abs(finite)) > 1e-4
    assert np.max(np.abs(analytic - finite)) < 1e-3


def test_fit_kernel_reduces_distance():
    target = jnp.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
        ]
    )
    result = fit_kernel(
        target,
        cost=_indicator_cost(3),
        gamma=0.9,
        x0=0,
        y0=0,
        steps=40,
        lr=0.2,
        key=jax.random.PRNGKey(0),
    )
    trace = result["distance_trace"]
    initial = float(trace[0])
    final = float(trace[-1])
    assert final < initial
    assert final < 0.5 * initial
    assert result["logits"].shape == (3, 3)


def test_gamma_zero_and_one_raise():
    kernel = jnp.eye(2)
    nu0 = jnp.array([1.0, 0.0])
    nu = occupancy(kernel, nu0, 0.5)
    cost = jnp.zeros((2, 2))
    lam = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    alpha = jnp.zeros((2, 2, 2))
    logits = jnp.zeros((2, 2))
    for gamma in (0.0, 1.0):
        with pytest.raises(ValueError):
            occupancy(kernel, nu0, gamma)
        with pytest.raises(ValueError):
            bisimulation_lp(nu, nu, cost, gamma, 0, 0)
        with pytest.raises(ValueError):
            envelope_grad(logits, lam, alpha, nu0, gamma)
        with pytest.raises(ValueError):
            fit_kernel(
                kernel,
                cost=cost,
                gamma=gamma,
                x0=0,
                y0=0,
                steps=1,
                lr=0.1,
                key=jax.random.PRNGKey(0),
            )
