"""Geometry tests for min-depth Chebyshev 3DPCS (eqs. 7-8)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_3dpcs.camera import project_pinhole
from jax_3dpcs.splat import splat_3dpcs, splat_projected


def _K(fx=50.0, fy=50.0, cx=16.0, cy=16.0):
    return jnp.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )


def test_optical_axis_chebyshev_disk_and_depth():
    H = W = 32
    cx = cy = 16
    R = 3
    z = 7.5
    K = _K(cx=float(cx), cy=float(cy))
    points = jnp.array([[0.0, 0.0, z]], dtype=jnp.float32)
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=R)

    uv, z_c, valid = project_pinhole(points, K)
    np.testing.assert_allclose(np.asarray(uv[0]), [cx, cy], atol=1e-5)
    assert bool(valid[0])
    np.testing.assert_allclose(np.asarray(z_c[0]), z, atol=1e-5)

    O = np.asarray(maps.O)
    D_final = np.asarray(maps.D_final)
    D_raw = np.asarray(maps.D_raw)

    rows, cols = np.indices((H, W))
    cheb = np.maximum(np.abs(cols - cx), np.abs(rows - cy))
    inside = cheb <= R
    np.testing.assert_array_equal(O[inside], 1.0)
    np.testing.assert_array_equal(O[~inside], 0.0)
    np.testing.assert_allclose(D_final[inside], z, atol=1e-5)
    np.testing.assert_array_equal(D_final[~inside], 0.0)
    np.testing.assert_allclose(D_raw[inside], z, atol=1e-5)
    assert np.all(np.isinf(D_raw[~inside]))

    # Chebyshev, not Euclidean: the square corner is occupied, the axis
    # pixel just beyond R is not.
    assert O[cy + R, cx + R] == 1.0
    assert O[cy, cx + R + 1] == 0.0
    assert O[cy + R + 1, cx] == 0.0


def test_two_points_same_pixel_keeps_nearer_depth():
    H = W = 32
    K = _K()
    near, far = 4.0, 11.0
    points = jnp.array(
        [[0.0, 0.0, near], [0.0, 0.0, far]], dtype=jnp.float32
    )
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=2)
    cx = cy = 16
    np.testing.assert_allclose(np.asarray(maps.D_final[cy, cx]), near, atol=1e-5)
    np.testing.assert_allclose(np.asarray(maps.D_raw[cy, cx]), near, atol=1e-5)
    # Whole overlapping disk is the nearer depth.
    O = np.asarray(maps.O)
    D = np.asarray(maps.D_final)
    np.testing.assert_allclose(D[O > 0], near, atol=1e-5)


def test_overlapping_disks_min_depth_in_intersection():
    H = W = 32
    K = _K(fx=50.0, fy=50.0, cx=16.0, cy=16.0)
    z1, z2 = 5.0, 3.0
    # Two camera-frame points that land 2 pixels apart horizontally.
    # u = fx * x / z + cx  =>  x = (u - cx) * z / fx
    p1 = jnp.array([0.0, 0.0, z1], dtype=jnp.float32)
    u2 = 18.0
    x2 = (u2 - 16.0) * z2 / 50.0
    p2 = jnp.array([x2, 0.0, z2], dtype=jnp.float32)
    maps = splat_3dpcs(jnp.stack([p1, p2]), K=K, height=H, width=W, radius=2)
    D = np.asarray(maps.D_final)
    # Pixel 16 is only in disk 1 (dist to 18 is 2, on the boundary of both:
    # dist(16,16)=0 for p1, dist(16,18)=2 for p2, both <= 2 so intersection
    # includes col 16..18. Min there is z2=3.
    np.testing.assert_allclose(D[16, 16], z2, atol=1e-4)
    np.testing.assert_allclose(D[16, 18], z2, atol=1e-4)


def test_behind_camera_ignored():
    H = W = 32
    K = _K()
    points = jnp.array([[0.0, 0.0, -5.0]], dtype=jnp.float32)
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=4)
    np.testing.assert_array_equal(np.asarray(maps.O), 0.0)
    np.testing.assert_array_equal(np.asarray(maps.D_final), 0.0)
    assert np.all(np.isinf(np.asarray(maps.D_raw)))


def test_outside_image_ignored():
    H = W = 32
    K = _K(fx=50.0, fy=50.0, cx=16.0, cy=16.0)
    # Far to the right: u = 50 * 20 / 2 + 16 = 516.
    points = jnp.array([[20.0, 0.0, 2.0]], dtype=jnp.float32)
    uv, _, valid = project_pinhole(points, K)
    assert bool(valid[0])
    assert float(uv[0, 0]) > W
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=3)
    np.testing.assert_array_equal(np.asarray(maps.O), 0.0)
    np.testing.assert_array_equal(np.asarray(maps.D_final), 0.0)


def test_empty_cloud():
    H = W = 16
    K = _K(cx=8.0, cy=8.0)
    points = jnp.zeros((0, 3), dtype=jnp.float32)
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=2)
    np.testing.assert_array_equal(np.asarray(maps.O), 0.0)
    np.testing.assert_array_equal(np.asarray(maps.D_final), 0.0)
    assert maps.D_final.shape == (H, W)
    assert maps.O.shape == (H, W)


def test_radius_zero_is_single_pixel():
    H = W = 16
    cx = cy = 8
    K = _K(cx=float(cx), cy=float(cy))
    points = jnp.array([[0.0, 0.0, 3.0]], dtype=jnp.float32)
    maps = splat_3dpcs(points, K=K, height=H, width=W, radius=0)
    O = np.asarray(maps.O)
    assert O.sum() == 1.0
    assert O[cy, cx] == 1.0
    np.testing.assert_allclose(np.asarray(maps.D_final[cy, cx]), 3.0, atol=1e-5)


def test_splat_projected_uses_valid_mask():
    uv = jnp.array([[4.0, 4.0], [4.0, 4.0]], dtype=jnp.float32)
    z_c = jnp.array([9.0, 2.0], dtype=jnp.float32)
    valid = jnp.array([True, False])
    maps = splat_projected(uv, z_c, valid, height=8, width=8, radius=1)
    np.testing.assert_allclose(np.asarray(maps.D_final[4, 4]), 9.0, atol=1e-5)
