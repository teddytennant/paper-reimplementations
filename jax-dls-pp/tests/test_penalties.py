"""Roughness penalties: constants have f1=0, lines have f2=0, Π ≻ 0."""

import jax.numpy as jnp

from dls_pp.estimate import constant_coefficients, linear_coefficients
from dls_pp.penalties import (
    f1,
    f2,
    is_positive_definite,
    omega_matrices,
    penalty_matrix,
    quadratic_form_penalty,
    roughness_penalty,
)
from dls_pp.splines import make_basis


def test_f1_vanishes_for_constant_trajectories():
    basis = make_basis(1.0, n_coeff=6)
    omega1, omega2 = omega_matrices(basis.knots, 1.0, n_quad=48)
    z = jnp.array([[0.4, -0.2], [1.1, 0.7]])
    c_x, c_y = constant_coefficients(z, basis.n_coeff)
    for i in range(z.shape[0]):
        assert abs(float(f1(c_x[i], c_y[i], omega1))) < 1e-8


def test_f2_vanishes_for_linear_trajectories():
    basis = make_basis(1.0, n_coeff=6)
    omega1, omega2 = omega_matrices(basis.knots, 1.0, n_quad=48)
    intercept = jnp.array([[0.0, 0.5], [-0.3, 0.2]])
    slope = jnp.array([[0.7, -0.4], [1.0, 0.1]])
    c_x, c_y = linear_coefficients(intercept, slope, basis.knots)
    for i in range(intercept.shape[0]):
        assert abs(float(f2(c_x[i], c_y[i], omega2))) < 1e-6


def test_penalty_matrix_is_positive_definite():
    basis = make_basis(1.0, n_coeff=5)
    omega1, omega2 = omega_matrices(basis.knots, 1.0, n_quad=32)
    pi = penalty_matrix(omega1, omega2, tau=1.0, rho1=0.2, rho2=0.05)
    assert bool(is_positive_definite(pi))


def test_quadratic_form_matches_roughness_penalty():
    basis = make_basis(1.0, n_coeff=5)
    omega1, omega2 = omega_matrices(basis.knots, 1.0, n_quad=32)
    tau, rho1, rho2 = 0.8, 0.15, 0.04
    pi = penalty_matrix(omega1, omega2, tau, rho1, rho2)
    intercept = jnp.array([[0.2, 0.0], [0.0, 0.3], [-0.1, 0.4]])
    slope = jnp.array([[0.1, 0.2], [-0.2, 0.0], [0.0, -0.1]])
    c_x, c_y = linear_coefficients(intercept, slope, basis.knots)
    p_rough = roughness_penalty(c_x, c_y, omega1, omega2, tau, rho1, rho2)
    p_quad = quadratic_form_penalty(c_x, c_y, pi)
    assert jnp.isclose(p_rough, p_quad, atol=1e-8)
