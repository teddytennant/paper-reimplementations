"""Anchoring, C0 projection, and a tiny clustered-network recovery check."""

import jax.numpy as jnp
from jax import random

from dls_pp.estimate import (
    apply_anchor,
    constant_coefficients,
    fit_dls_pp,
    mean_pairwise_intensity,
    project_positions_c0,
    simulate_network,
)
from dls_pp.intensity import apply_rigid_transform, pairwise_squared_distances
from dls_pp.splines import make_basis


def test_apply_rigid_transform_changes_coords_preserves_distances():
    z = jnp.array([[0.2, 0.5], [1.0, -0.3], [-0.4, 0.8], [0.6, 0.1]])
    translation = jnp.array([0.25, -0.4])
    theta = 1.1
    rotation = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    z2 = apply_rigid_transform(z, translation, rotation)
    assert not jnp.allclose(z2, z)
    assert jnp.allclose(
        pairwise_squared_distances(z2), pairwise_squared_distances(z), atol=1e-10
    )


def test_project_positions_c0_puts_node1_at_origin_node2_on_x_axis():
    z = jnp.array([[2.0, 1.0], [3.5, 2.5], [0.0, 4.0], [1.0, 0.0]])
    zp = project_positions_c0(z)
    assert jnp.allclose(zp[0], jnp.zeros(2), atol=1e-10)
    assert abs(float(zp[1, 1])) < 1e-10
    assert jnp.allclose(
        pairwise_squared_distances(zp), pairwise_squared_distances(z), atol=1e-10
    )


def test_apply_anchor_zeros_required_coefficients():
    c_x = jnp.ones((4, 5))
    c_y = jnp.ones((4, 5))
    c_x, c_y = apply_anchor(c_x, c_y)
    assert jnp.allclose(c_x[0], 0.0)
    assert jnp.allclose(c_y[0], 0.0)
    assert jnp.allclose(c_y[1], 0.0)


def test_fit_recovers_higher_intensity_for_close_dyads():
    n = 6
    t_max = 1.0
    n_coeff = 4
    z_true = jnp.array(
        [
            [-0.85, 0.05],
            [-0.75, -0.08],
            [-0.80, 0.02],
            [0.82, 0.04],
            [0.78, -0.06],
            [0.85, 0.01],
        ]
    )
    beta_true = jnp.full((n,), 0.6)
    basis = make_basis(t_max, n_coeff)
    c_x, c_y = constant_coefficients(z_true, n_coeff)
    key = random.PRNGKey(7)
    events = simulate_network(key, beta_true, c_x, c_y, basis.knots, t_max, model="full")
    assert events.n_events > 0

    result = fit_dls_pp(
        events,
        n_coeff=n_coeff,
        n_steps=200,
        batch_size=15,
        lr=0.08,
        tau=1.0,
        rho1=0.05,
        rho2=0.01,
        n_quad=16,
        model="full",
        seed=3,
        beta_every=10,
    )
    times = jnp.linspace(0.0, t_max, 8)
    mean_lam = mean_pairwise_intensity(
        result.beta, result.c_x, result.c_y, times, basis.knots, model="full"
    )

    close_pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    far_pairs = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    close_mean = float(jnp.mean(jnp.array([mean_lam[i, j] for i, j in close_pairs])))
    far_mean = float(jnp.mean(jnp.array([mean_lam[i, j] for i, j in far_pairs])))
    assert close_mean > far_mean
