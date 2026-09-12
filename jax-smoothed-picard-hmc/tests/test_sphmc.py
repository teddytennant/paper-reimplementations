"""CPU checks of Smoothed Picard HMC identities from arXiv:2609.06906."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sphmc.kernel import make_quadratic_score, picard_hmc_step
from sphmc.prox import prox_gradient_descent, prox_quadratic
from sphmc.quadrature import chebyshev_lobatto_nodes, quadrature_weights
from sphmc.sampler import SPHMCConfig, sample_quadratic


def _random_poly_coeffs(rng: np.random.Generator, degree: int) -> np.ndarray:
    return rng.normal(size=degree + 1)


def _eval_poly(coeff: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.polyval(coeff, t)


def _int_f(coeff: np.ndarray, a: float, b: float) -> float:
    n = coeff.shape[0]
    powers = np.arange(n, 0, -1, dtype=np.float64)
    anti = np.append(coeff / powers, 0.0)
    return float(np.polyval(anti, b) - np.polyval(anti, a))


def _int_ti_minus_t_f(coeff: np.ndarray, ti: float) -> float:
    times_t = np.polymul(coeff, np.array([1.0, 0.0], dtype=np.float64))
    return ti * _int_f(coeff, 0.0, ti) - _int_f(times_t, 0.0, ti)


@pytest.mark.parametrize("J", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("h", [0.1, 0.7, 1.3])
def test_quadrature_identities_on_polynomials(J: int, h: float):
    rng = np.random.default_rng(J * 100 + int(10 * h))
    quad = quadrature_weights(J, h)
    nodes = quad.t
    for _ in range(6):
        coeff = _random_poly_coeffs(rng, J - 1)
        f_nodes = _eval_poly(coeff, nodes)
        for i in range(J):
            ti = float(nodes[i])
            expected = _int_ti_minus_t_f(coeff, ti)
            got = float(quad.omega[i] @ f_nodes)
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-10)
        expected_end = _int_f(coeff, 0.0, h)
        got_end = float(quad.omega_end @ f_nodes)
        assert got_end == pytest.approx(expected_end, rel=1e-9, abs=1e-10)
        for i in range(J):
            ti = float(nodes[i])
            expected_p = _int_f(coeff, 0.0, ti)
            got_p = float(quad.omega_prime[i] @ f_nodes)
            assert got_p == pytest.approx(expected_p, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("J", [2, 3, 4, 5, 8])
def test_omega_nonnegative_and_sum_to_h(J: int):
    h = 0.37
    quad = quadrature_weights(J, h)
    assert np.all(quad.omega_end >= -1e-12)
    assert float(np.sum(quad.omega_end)) == pytest.approx(h, rel=1e-12, abs=1e-12)
    t = chebyshev_lobatto_nodes(J, h)
    assert t[0] == pytest.approx(0.0, abs=1e-15)
    assert t[-1] == pytest.approx(h, abs=1e-12)
    assert np.all(np.diff(t) >= -1e-15)


def test_d2_agrees_with_closed_form_prox():
    rng = np.random.default_rng(7)
    d = 4
    A_np = np.diag(np.array([0.4, 0.8, 1.2, 1.6]))
    A = jnp.asarray(A_np)
    eta = 0.05

    def grad_fn(x):
        return A @ x

    for _ in range(8):
        y = jnp.asarray(rng.normal(size=d))
        closed = prox_quadratic(y, A, eta)
        gd = prox_gradient_descent(y, grad_fn, eta, eps_prox=1e-8, max_iter=4096)
        np.testing.assert_allclose(np.asarray(gd), np.asarray(closed), atol=1e-5, rtol=1e-5)


def test_picard_left_endpoint_is_fixed():
    d = 3
    J = 5
    h = 0.2
    eta = 0.05
    n_picard = 2
    A = jnp.diag(jnp.array([0.5, 1.0, 1.5]))
    quad = quadrature_weights(J, h)
    score_fn = make_quadratic_score(A, eta)
    key = jax.random.PRNGKey(3)
    x = jnp.array([0.2, -0.4, 0.1])
    p = jnp.array([0.3, 0.1, -0.2])
    state = picard_hmc_step(key, x, p, score_fn, quad, n_picard)
    np.testing.assert_allclose(np.asarray(state.x_nodes[0]), np.asarray(x), atol=1e-10)
    np.testing.assert_allclose(np.asarray(state.p_nodes[0]), np.asarray(p), atol=1e-10)
    assert float(quad.t[0]) == pytest.approx(0.0, abs=1e-15)


def _smoothed_hamiltonian(x, p, eta: float) -> float:
    # V_η for V(x)=1/2 ||x||^2 is 1/(2(1+η)) ||x||^2; kinetic energy 1/2 ||p||^2.
    return 0.5 * float(jnp.sum(p**2)) + 0.5 / (1.0 + eta) * float(jnp.sum(x**2))


def test_smoothed_energy_drift_bounded_for_small_h():
    d = 2
    J = 5
    h = 0.05
    eta = 0.05
    n_picard = 2
    A = jnp.eye(d)
    quad = quadrature_weights(J, h)
    score_fn = make_quadratic_score(A, eta)
    key = jax.random.PRNGKey(11)
    x = jnp.array([0.4, -0.3])
    p = jnp.array([0.2, 0.5])
    H0 = _smoothed_hamiltonian(x, p, eta)
    drifts = []
    for i in range(40):
        key, k_step = jax.random.split(key)
        state = picard_hmc_step(k_step, x, p, score_fn, quad, n_picard)
        x, p = state.x, state.p
        drifts.append(abs(_smoothed_hamiltonian(x, p, eta) - H0))
    assert max(drifts) < 0.35
    assert float(np.mean(drifts)) < 0.15


def test_sampler_standard_gaussian_mean_and_variance():
    key = jax.random.PRNGKey(0)
    A = jnp.eye(2)
    config = SPHMCConfig(h=0.1, n_picard=1, J=4, eta=0.05, gamma=1.0)
    samples = sample_quadratic(
        key, A, config, n_steps=40, n_samples=256, recover_rgo=True
    )
    mean = np.asarray(jnp.mean(samples, axis=0))
    var = np.asarray(jnp.var(samples, axis=0))
    # Independent chains; mean should be O(1/sqrt(n)).
    assert np.max(np.abs(mean)) < 0.25
    assert float(np.mean(np.abs(var - 1.0))) < 0.5


def test_ill_conditioned_quadratic_weak_axis_has_larger_variance():
    key = jax.random.PRNGKey(21)
    alpha, beta = 0.2, 2.0
    A = jnp.diag(jnp.array([alpha, beta]))
    config = SPHMCConfig(h=0.08, n_picard=2, J=5, eta=0.05, gamma=1.0)
    samples = sample_quadratic(
        key, A, config, n_steps=50, n_samples=384, recover_rgo=True
    )
    var = np.asarray(jnp.var(samples, axis=0))
    # Target covariance is A^{-1} = diag(1/α, 1/β) = (5, 0.5).
    assert var[0] > var[1]
    assert var[0] > 1.5
    assert var[1] < 1.5
