"""Intensity equation 2.1 and nested-model restrictions."""

import jax.numpy as jnp

from dls_pp.intensity import (
    apply_rigid_transform,
    intensity,
    log_intensity,
    pairwise_log_intensity,
    pairwise_squared_distances,
)
from dls_pp.estimate import constant_coefficients
from dls_pp.splines import evaluate_trajectories, make_basis


def test_eq_2_1_unit_distance_gives_exp_minus_one():
    z_i = jnp.array([0.0, 0.0])
    z_j = jnp.array([1.0, 0.0])
    lam = intensity(0.0, 0.0, z_i, z_j, model="full")
    assert jnp.isclose(lam, jnp.exp(-1.0), atol=1e-12)


def test_eq_2_1_log_intensity_beta_and_distance():
    # β_i=0.5, β_j=-0.2, z_i=(0,1), z_j=(0,0) → log λ = 0.3 − 1 = −0.7
    z_i = jnp.array([0.0, 1.0])
    z_j = jnp.array([0.0, 0.0])
    log_lam = log_intensity(0.5, -0.2, z_i, z_j, model="full")
    assert jnp.isclose(log_lam, -0.7, atol=1e-12)


def test_beta_only_independent_of_positions():
    beta_i, beta_j = 0.4, -0.1
    z_a_i = jnp.array([0.0, 0.0])
    z_a_j = jnp.array([3.0, 1.0])
    z_b_i = jnp.array([1.5, -2.0])
    z_b_j = jnp.array([0.2, 0.2])
    lam_a = intensity(beta_i, beta_j, z_a_i, z_a_j, model="beta_only")
    lam_b = intensity(beta_i, beta_j, z_b_i, z_b_j, model="beta_only")
    assert jnp.isclose(lam_a, lam_b, atol=1e-12)
    assert jnp.isclose(lam_a, jnp.exp(beta_i + beta_j), atol=1e-12)


def test_homogeneous_constant_in_t_when_coefficients_constant():
    z = jnp.array([[0.0, 0.0], [1.0, 0.5], [-0.3, 0.8]])
    beta = jnp.array([0.1, -0.2, 0.0])
    basis = make_basis(1.0, n_coeff=5)
    c_x, c_y = constant_coefficients(z, basis.n_coeff)
    times = jnp.linspace(0.0, 1.0, 11)
    traj = evaluate_trajectories(c_x, c_y, times, basis.knots)
    log_lams = []
    for m in range(times.shape[0]):
        log_lams.append(log_intensity(beta[0], beta[1], traj[0, m], traj[1, m], model="full"))
    stacked = jnp.stack(log_lams)
    assert jnp.allclose(stacked, stacked[0], atol=1e-10)


def test_pairwise_matches_scalar_and_rigid_transform_preserves_distances():
    z = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    beta = jnp.array([0.0, 0.1, -0.2])
    d2 = pairwise_squared_distances(z)
    assert jnp.isclose(d2[0, 1], 1.0)
    log_pw = pairwise_log_intensity(beta, z, model="full")
    assert jnp.isclose(log_pw[0, 1], log_intensity(beta[0], beta[1], z[0], z[1]))

    translation = jnp.array([0.3, -0.7])
    theta = 0.4
    rotation = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    z_rot = apply_rigid_transform(z, translation, rotation)
    assert not jnp.allclose(z_rot, z)
    assert jnp.allclose(pairwise_squared_distances(z_rot), d2, atol=1e-10)
    assert jnp.allclose(
        pairwise_log_intensity(beta, z_rot, model="full"), log_pw, atol=1e-10
    )
