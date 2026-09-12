"""Poisson log-likelihood, closed form, and nested homogeneous model."""

import jax.numpy as jnp

from dls_pp.estimate import constant_coefficients
from dls_pp.intensity import intensity
from dls_pp.likelihood import (
    Events,
    closed_form_constant_loglik,
    dyad_indices,
    homogeneous_loglik,
    poisson_loglik,
    quadrature_grid,
)
from dls_pp.splines import make_basis


def _two_node_events(times, t_max=1.0):
    n = int(times.shape[0])
    return Events(
        src=jnp.zeros((n,), dtype=jnp.int32),
        dst=jnp.ones((n,), dtype=jnp.int32),
        times=jnp.asarray(times, dtype=jnp.float64),
        n_nodes=2,
        t_max=float(t_max),
    )


def test_poisson_loglik_matches_closed_form_constant_lambda():
    t_max = 1.0
    times = jnp.array([0.05, 0.2, 0.35, 0.55, 0.8, 0.95])
    events = _two_node_events(times, t_max=t_max)
    z = jnp.array([[0.0, 0.0], [0.5, 0.0]])
    beta = jnp.array([0.0, 0.0])
    lam = float(intensity(beta[0], beta[1], z[0], z[1], model="full"))
    closed = closed_form_constant_loglik(events.n_events, lam, t_max)

    basis = make_basis(t_max, n_coeff=4)
    c_x, c_y = constant_coefficients(z, basis.n_coeff)
    quad_t, quad_w = quadrature_grid(t_max, n_quad=16)
    ll = poisson_loglik(beta, c_x, c_y, events, basis.knots, quad_t, quad_w, model="full")
    assert abs(float(ll) - closed) < 1e-4


def test_homogeneous_loglik_agrees_with_poisson_for_constant_z():
    t_max = 1.2
    times = jnp.array([0.1, 0.4, 0.9])
    events = _two_node_events(times, t_max=t_max)
    z = jnp.array([[0.1, -0.2], [0.4, 0.3]])
    beta = jnp.array([0.2, -0.1])
    basis = make_basis(t_max, n_coeff=4)
    c_x, c_y = constant_coefficients(z, basis.n_coeff)
    quad_t, quad_w = quadrature_grid(t_max, n_quad=20)
    ll_p = poisson_loglik(beta, c_x, c_y, events, basis.knots, quad_t, quad_w, model="full")
    ll_h = homogeneous_loglik(beta, z, events, t_max, model="homogeneous")
    assert abs(float(ll_p) - float(ll_h)) < 1e-4


def test_dyad_indices_are_upper_triangle():
    src, dst = dyad_indices(4)
    assert src.shape[0] == 6
    assert jnp.all(src < dst)
