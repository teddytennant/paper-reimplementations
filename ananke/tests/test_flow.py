"""Tests for the exact circular flow and the CTAN-Res smoke model."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from ananke.block import ctan_res, init_ctan_res, init_tiny_classifier, tiny_classifier, train_tiny_classifier
from ananke.flow import (
    apply_dual_shear,
    exact_circular_flow,
    flow_quantities,
    flow_vs_rk4_error,
    product_flow,
    rotation_matrix,
)


ATOL = 1e-10
DTYPE = jnp.float64


def _f(*values):
    return [jnp.asarray(v, dtype=DTYPE) for v in values]


def test_rotation_matrix_matches_action():
    theta = jnp.asarray(0.7, dtype=DTYPE)
    v = jnp.array([0.3, -1.2], dtype=DTYPE)
    rot = rotation_matrix(theta)
    applied = rot @ v
    manual = jnp.array(
        [
            jnp.cos(theta) * v[0] - jnp.sin(theta) * v[1],
            jnp.sin(theta) * v[0] + jnp.cos(theta) * v[1],
        ],
        dtype=DTYPE,
    )
    assert jnp.allclose(applied, manual, atol=ATOL)
    assert jnp.allclose(jnp.linalg.det(rot), jnp.asarray(1.0, dtype=DTYPE), atol=ATOL)


@pytest.mark.parametrize("t", [0.0, 0.25, 1.0, 2.5])
@pytest.mark.parametrize("beta", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("radius", [0.5, 1.0, 2.5])
def test_log_radius_decays_exactly(t, beta, radius):
    c, = _f([0.2, -0.4])
    # Several radii relative to R, including a batch.
    offsets = jnp.array(
        [
            [1.7, 0.0],
            [0.2, -0.3],
            [-0.8, 0.5],
            [0.0, radius],
        ],
        dtype=DTYPE,
    )
    x = c + offsets
    alpha, twist = _f(0.4, -0.7)
    q = flow_quantities(x, c, radius, beta, alpha, twist, t)
    r0 = jnp.linalg.norm(x - c, axis=-1)
    z0 = jnp.log(r0 / radius)
    expected_z = jnp.exp(-beta * t) * z0
    assert jnp.allclose(q["z"], expected_z, atol=ATOL)
    assert jnp.allclose(q["q"], jnp.exp(-beta * t), atol=ATOL)
    geometric_z = jnp.log(jnp.linalg.norm(q["y"] - c, axis=-1) / radius)
    assert jnp.allclose(geometric_z, expected_z, atol=ATOL)


@pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("angle0", [0.0, 0.7, -1.2, 2.5])
def test_circle_is_invariant_and_t0_is_identity(t, angle0):
    c, radius, beta, alpha, twist = _f([0.3, -0.2], 1.25, 0.9, 1.1, 0.4)
    direction = jnp.array([jnp.cos(angle0), jnp.sin(angle0)], dtype=DTYPE)
    direction = direction / jnp.linalg.norm(direction)
    x = c + radius * direction
    y = exact_circular_flow(x, c, radius, beta, alpha, twist, t)
    out_r = jnp.linalg.norm(y - c)
    assert jnp.allclose(out_r, radius, atol=ATOL)
    y0 = exact_circular_flow(x, c, radius, beta, alpha, twist, 0.0)
    assert jnp.allclose(y0, x, atol=ATOL)


def test_t0_identity_off_circle():
    x, c = _f([1.4, -0.6], [0.2, 0.5])
    y = exact_circular_flow(x, c, 0.8, 1.3, -0.4, 0.6, 0.0)
    assert jnp.allclose(y, x, atol=ATOL)


@pytest.mark.parametrize("t", [0.0, 0.4, 1.2])
@pytest.mark.parametrize("beta", [0.5, 1.7])
def test_rho_and_radius_formulas(t, beta):
    c = jnp.array([0.0, 0.0], dtype=DTYPE)
    x = jnp.array([1.3, -0.7], dtype=DTYPE)
    radius, alpha, twist = _f(0.9, 0.2, 0.3)
    q = flow_quantities(x, c, radius, beta, alpha, twist, t)
    r0 = jnp.linalg.norm(x - c)
    z0 = jnp.log(r0 / radius)
    contraction = jnp.exp(-beta * t)
    expected_rho = (radius / r0) ** (1.0 - contraction)
    expected_r = radius * jnp.exp(contraction * z0)
    assert jnp.allclose(q["rho"], expected_rho, atol=ATOL)
    assert jnp.allclose(q["r"], expected_r, atol=ATOL)
    assert jnp.allclose(q["rho"] * r0, expected_r, atol=ATOL)
    assert jnp.allclose(jnp.linalg.norm(q["y"] - c), expected_r, atol=ATOL)


def test_theta_matches_equation_including_zero_twist():
    c = jnp.array([-0.2, 0.4], dtype=DTYPE)
    x = c + jnp.array([1.5, 0.25], dtype=DTYPE)
    radius = jnp.asarray(0.7, dtype=DTYPE)
    beta = jnp.asarray(1.1, dtype=DTYPE)
    alpha = jnp.asarray(0.55, dtype=DTYPE)
    t = jnp.asarray(1.3, dtype=DTYPE)
    r0 = jnp.linalg.norm(x - c)
    z0 = jnp.log(r0 / radius)
    contraction = jnp.exp(-beta * t)

    for twist_value in (0.0, 0.8, -1.4):
        twist = jnp.asarray(twist_value, dtype=DTYPE)
        q = flow_quantities(x, c, radius, beta, alpha, twist, t)
        expected = alpha * t + (twist / beta) * (1.0 - contraction) * z0
        assert jnp.allclose(q["theta"], expected, atol=ATOL)
        if twist_value == 0.0:
            assert jnp.allclose(q["theta"], alpha * t, atol=ATOL)


def test_explicit_twist_point():
    # r0 = e, R = 1, z0 = 1, beta = 1, t = log(2), q = 1/2, a = 2, alpha = 0
    # theta = 1, r = exp(1/2)
    radius = jnp.asarray(1.0, dtype=DTYPE)
    beta = jnp.asarray(1.0, dtype=DTYPE)
    t = jnp.log(jnp.asarray(2.0, dtype=DTYPE))
    x = jnp.array([jnp.exp(jnp.asarray(1.0, dtype=DTYPE)), 0.0], dtype=DTYPE)
    c = jnp.zeros(2, dtype=DTYPE)
    q = flow_quantities(x, c, radius, beta, 0.0, 2.0, t)
    assert jnp.allclose(q["theta"], jnp.asarray(1.0, dtype=DTYPE), atol=ATOL)
    assert jnp.allclose(q["r"], jnp.exp(jnp.asarray(0.5, dtype=DTYPE)), atol=ATOL)
    expected = q["r"] * jnp.array([jnp.cos(1.0), jnp.sin(1.0)], dtype=DTYPE)
    assert jnp.allclose(q["y"], expected, atol=ATOL)


def test_analytic_closer_to_fine_rk4_than_coarse():
    x = jnp.array([1.7, -0.4], dtype=DTYPE)
    c = jnp.array([0.2, 0.1], dtype=DTYPE)
    radius = jnp.asarray(1.0, dtype=DTYPE)
    beta = jnp.asarray(0.8, dtype=DTYPE)
    alpha = jnp.asarray(1.2, dtype=DTYPE)
    twist = jnp.asarray(0.5, dtype=DTYPE)
    t = 1.0
    err_fine = flow_vs_rk4_error(x, c, radius, beta, alpha, twist, t, dt=1e-3)
    # Coarse trajectory vs the same fine RK4, in Cartesian coordinates.
    from ananke.flow import cartesian_from_polar, integrate_polar_rk4

    delta = x - c
    r0 = jnp.linalg.norm(delta)
    theta0 = jnp.arctan2(delta[1], delta[0])
    r_fine, th_fine = integrate_polar_rk4(r0, theta0, radius, beta, alpha, twist, t, 1e-3)
    r_coarse, th_coarse = integrate_polar_rk4(r0, theta0, radius, beta, alpha, twist, t, 0.25)
    y_fine = cartesian_from_polar(c, r_fine, th_fine)
    y_coarse = cartesian_from_polar(c, r_coarse, th_coarse)
    err_coarse = jnp.max(jnp.abs(y_coarse - y_fine))
    assert float(err_fine) < 1e-6
    assert float(err_fine) < float(err_coarse)


@pytest.mark.parametrize(
    "r0_scale,twist",
    [(0.4, 0.0), (1.0, 0.3), (2.2, -0.6), (1.5, 1.1)],
)
def test_rk4_agreement_several_ic(r0_scale, twist):
    c = jnp.array([-0.15, 0.25], dtype=DTYPE)
    radius = jnp.asarray(0.8, dtype=DTYPE)
    direction = jnp.array([0.6, -0.8], dtype=DTYPE)
    direction = direction / jnp.linalg.norm(direction)
    x = c + (r0_scale * radius) * direction
    err = flow_vs_rk4_error(x, c, radius, 1.1, -0.4, twist, t=1.0, dt=1e-3)
    assert float(err) < 1e-6


def test_product_structure_matches_per_plane():
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    batch = 5
    planes = 3
    x = jax.random.normal(k1, (batch, 2 * planes), dtype=DTYPE)
    c = jax.random.normal(k2, (planes, 2), dtype=DTYPE)
    radius = jnp.array([0.5, 1.0, 1.4], dtype=DTYPE)
    beta = jnp.array([0.4, 0.8, 1.5], dtype=DTYPE)
    alpha = jnp.array([0.2, -0.3, 0.7], dtype=DTYPE)
    twist = jnp.array([0.0, 0.5, -0.2], dtype=DTYPE)
    t = jnp.asarray(0.75, dtype=DTYPE)
    y = product_flow(x, c, radius, beta, alpha, twist, t)
    for i in range(planes):
        yi = exact_circular_flow(
            x[:, 2 * i : 2 * i + 2],
            c[i],
            radius[i],
            beta[i],
            alpha[i],
            twist[i],
            t,
        )
        assert jnp.allclose(y[:, 2 * i : 2 * i + 2], yi, atol=ATOL)
    # Shared scalar parameters: plane-wise equals the product map.
    y_shared = product_flow(x, jnp.zeros(2, dtype=DTYPE), 1.0, 0.6, 0.3, 0.1, 0.5)
    chunks = []
    for i in range(planes):
        chunks.append(
            exact_circular_flow(
                x[:, 2 * i : 2 * i + 2],
                jnp.zeros(2, dtype=DTYPE),
                1.0,
                0.6,
                0.3,
                0.1,
                0.5,
            )
        )
    assert jnp.allclose(y_shared, jnp.concatenate(chunks, axis=-1), atol=ATOL)
    assert y.shape == x.shape
    # Silence unused key in case of future edits.
    assert k3.shape == (2,)


def test_diffeomorphism_jacobian_nonzero():
    c = jnp.array([0.1, -0.2], dtype=DTYPE)
    radius = jnp.asarray(1.0, dtype=DTYPE)
    beta = jnp.asarray(0.7, dtype=DTYPE)
    alpha = jnp.asarray(0.4, dtype=DTYPE)
    twist = jnp.asarray(0.25, dtype=DTYPE)
    x = jnp.array([1.4, 0.3], dtype=DTYPE)
    fd = jnp.asarray(1e-6, dtype=DTYPE)

    def phi(v, t):
        return exact_circular_flow(v, c, radius, beta, alpha, twist, t)

    for t in (0.0, 0.3, 1.0, 2.0):
        cols = []
        for i in range(2):
            e = jnp.zeros(2, dtype=DTYPE).at[i].set(fd)
            cols.append((phi(x + e, t) - phi(x - e, t)) / (2.0 * fd))
        jac = jnp.stack(cols, axis=-1)
        det = jnp.linalg.det(jac)
        assert jnp.isfinite(det)
        assert abs(float(det)) > 1e-5
        step = jnp.array([1e-4, -2e-4], dtype=DTYPE)
        out_step = phi(x + step, t) - phi(x, t)
        assert jnp.all(jnp.isfinite(out_step))
        assert float(jnp.linalg.norm(out_step)) < 50.0 * float(jnp.linalg.norm(step))


def test_dual_shear_det_and_fma():
    s = jnp.asarray(0.4, dtype=DTYPE)
    v = jnp.array([0.8, -0.3], dtype=DTYPE)
    y = apply_dual_shear(v, s)
    matrix = jnp.array([[1.0 - s * s, -s], [s, 1.0]], dtype=DTYPE)
    assert jnp.allclose(y, matrix @ v, atol=ATOL)
    assert jnp.allclose(jnp.linalg.det(matrix), jnp.asarray(1.0, dtype=DTYPE), atol=ATOL)


def test_ctan_res_shape_and_gamma_zero_identity():
    key = jax.random.PRNGKey(0)
    channels = 8
    params = init_ctan_res(key, channels, dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, channels, 5, 7), dtype=jnp.float32)
    y = ctan_res(params, x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))

    zero_gamma = dict(params)
    zero_gamma["gamma"] = jnp.zeros_like(params["gamma"])
    y_id = ctan_res(zero_gamma, x)
    assert jnp.allclose(y_id, x, atol=0.0, rtol=0.0)

    # Odd spatial size and a second batch, still identity at gamma = 0.
    x2 = jax.random.normal(jax.random.PRNGKey(2), (1, channels, 8, 8), dtype=jnp.float32)
    assert ctan_res(zero_gamma, x2).shape == x2.shape
    assert jnp.allclose(ctan_res(zero_gamma, x2), x2, atol=0.0, rtol=0.0)


def test_tiny_classifier_forward():
    params = init_tiny_classifier(jax.random.PRNGKey(0), width=8, n_classes=4)
    images = jax.random.normal(jax.random.PRNGKey(1), (4, 3, 8, 8), dtype=jnp.float32)
    logits = tiny_classifier(params, images)
    assert logits.shape == (4, 4)
    assert jnp.all(jnp.isfinite(logits))


def test_tiny_train_loss_decreases():
    loss0, loss20 = train_tiny_classifier(steps=20, seed=0)
    assert jnp.isfinite(loss0)
    assert jnp.isfinite(loss20)
    assert float(loss20) < float(loss0)
