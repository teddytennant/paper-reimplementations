"""Clamped cubic B-splines: affine reproduction and eq 2.6 quadratic distances."""

import jax.numpy as jnp

from dls_pp.estimate import linear_coefficients
from dls_pp.splines import (
    SPLINE_DEGREE,
    basis_vector,
    evaluate_trajectories,
    greville_abscissae,
    make_basis,
    squared_distance_quadratic,
)


def test_spline_degree_is_cubic():
    assert SPLINE_DEGREE == 3


def test_clamped_cubics_reproduce_affine_functions():
    t_max = 1.5
    basis = make_basis(t_max, n_coeff=7)
    intercept = jnp.array([[0.2, -0.4], [1.0, 0.5], [-0.7, 0.3]])
    slope = jnp.array([[0.8, -0.1], [0.0, 1.2], [0.4, -0.6]])
    c_x, c_y = linear_coefficients(intercept, slope, basis.knots)
    times = jnp.linspace(0.0, t_max, 25)
    traj = evaluate_trajectories(c_x, c_y, times, basis.knots)
    expected = intercept[:, None, :] + slope[:, None, :] * times[None, :, None]
    assert jnp.allclose(traj, expected, atol=1e-8)


def test_greville_abscissae_are_coefficients_of_identity():
    basis = make_basis(2.0, n_coeff=6)
    grev = greville_abscissae(basis.knots)
    times = jnp.linspace(0.0, 2.0, 17)
    # φ(t) · greville = t for clamped cubics
    values = jnp.stack([basis_vector(t, basis.knots) @ grev for t in times])
    assert jnp.allclose(values, times, atol=1e-8)


def test_eq_2_6_quadratic_matches_trajectory_distance():
    basis = make_basis(1.0, n_coeff=6)
    intercept = jnp.array([[0.0, 0.0], [1.0, 0.2], [-0.5, 0.8]])
    slope = jnp.array([[0.3, 0.1], [-0.2, 0.4], [0.0, -0.3]])
    c_x, c_y = linear_coefficients(intercept, slope, basis.knots)
    times = jnp.linspace(0.0, 1.0, 13)
    traj = evaluate_trajectories(c_x, c_y, times, basis.knots)
    for m, t in enumerate(times):
        phi = basis_vector(t, basis.knots)
        d2_quad = squared_distance_quadratic(c_x[0], c_x[1], c_y[0], c_y[1], phi)
        diff = traj[0, m] - traj[1, m]
        d2_traj = jnp.sum(diff * diff)
        assert jnp.isclose(d2_quad, d2_traj, atol=1e-10)
