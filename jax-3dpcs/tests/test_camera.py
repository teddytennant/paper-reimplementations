"""Projection geometry against hand-computed pinhole pixels (eq. 7)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_3dpcs.camera import camera_matrix, project_P, project_pinhole


def _intrinsics(fx=100.0, fy=80.0, cx=32.0, cy=24.0):
    return jnp.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )


def test_optical_axis_hits_principal_point():
    K = _intrinsics(cx=32.0, cy=24.0)
    z = 5.0
    points = jnp.array([[0.0, 0.0, z]], dtype=jnp.float32)
    uv, z_c, valid = project_pinhole(points, K)
    np.testing.assert_allclose(np.asarray(uv[0]), [32.0, 24.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(z_c[0]), z, atol=1e-5)
    assert bool(valid[0])


def test_known_grid_matches_hand_pixels():
    fx, fy, cx, cy = 100.0, 80.0, 32.0, 24.0
    K = _intrinsics(fx, fy, cx, cy)
    xs = jnp.array([-1.0, 0.0, 1.5], dtype=jnp.float32)
    ys = jnp.array([-0.5, 0.25], dtype=jnp.float32)
    zs = jnp.array([2.0, 4.0, 8.0], dtype=jnp.float32)
    xx, yy, zz = jnp.meshgrid(xs, ys, zs, indexing="ij")
    points = jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    uv, z_c, valid = project_pinhole(points, K)
    expected_u = fx * points[:, 0] / points[:, 2] + cx
    expected_v = fy * points[:, 1] / points[:, 2] + cy

    np.testing.assert_allclose(np.asarray(uv[:, 0]), np.asarray(expected_u), atol=1e-5)
    np.testing.assert_allclose(np.asarray(uv[:, 1]), np.asarray(expected_v), atol=1e-5)
    np.testing.assert_allclose(np.asarray(z_c), np.asarray(points[:, 2]), atol=1e-5)
    assert bool(np.all(np.asarray(valid)))


def test_extrinsic_rotation_and_translation():
    K = _intrinsics()
    # 90 deg about z: (x, y, z) -> (-y, x, z)
    R = jnp.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    t = jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32)
    p_l = jnp.array([[1.0, 2.0, 4.0]], dtype=jnp.float32)
    p_c = (R @ p_l[0] + t)
    uv, z_c, valid = project_pinhole(p_l, K, R=R, t=t)
    expected_u = 100.0 * p_c[0] / p_c[2] + 32.0
    expected_v = 80.0 * p_c[1] / p_c[2] + 24.0
    np.testing.assert_allclose(np.asarray(uv[0, 0]), np.asarray(expected_u), atol=1e-5)
    np.testing.assert_allclose(np.asarray(uv[0, 1]), np.asarray(expected_v), atol=1e-5)
    np.testing.assert_allclose(np.asarray(z_c[0]), np.asarray(p_c[2]), atol=1e-5)
    assert bool(valid[0])


def test_P_matches_KRt():
    K = _intrinsics()
    R = jnp.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    t = jnp.array([0.2, 0.1, -0.3], dtype=jnp.float32)
    P = camera_matrix(K, R, t)
    points = jnp.array(
        [[0.0, 0.0, 5.0], [1.0, -1.0, 3.0], [-0.4, 0.7, 6.0]], dtype=jnp.float32
    )
    uv_k, z_k, v_k = project_pinhole(points, K, R=R, t=t)
    uv_p, z_p, v_p = project_P(points, P)
    np.testing.assert_allclose(np.asarray(uv_k), np.asarray(uv_p), atol=1e-5)
    np.testing.assert_allclose(np.asarray(z_k), np.asarray(z_p), atol=1e-5)
    np.testing.assert_array_equal(np.asarray(v_k), np.asarray(v_p))


def test_behind_camera_is_invalid():
    K = _intrinsics()
    points = jnp.array([[0.0, 0.0, -3.0], [1.0, 1.0, 0.0]], dtype=jnp.float32)
    uv, z_c, valid = project_pinhole(points, K)
    assert not bool(valid[0])
    assert not bool(valid[1])
    np.testing.assert_allclose(np.asarray(uv), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z_c), np.array([-3.0, 0.0]), atol=1e-5)
