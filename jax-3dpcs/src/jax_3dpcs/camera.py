"""Virtual pinhole projection for 3DPCS (PRI-Net eq. 7).

Paper eq. 7 maps a LiDAR point p^l through a virtual camera (K^v, R^v, t^v):

    ũ^v = (1 / z^c) K^v (R^v p^l + t^v)

with z^c the camera-frame depth (third coordinate of R^v p^l + t^v).
The 3x4 matrix P = K^v [R^v | t^v] is an equivalent encoding of the same map.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def camera_matrix(K: Array, R: Array, t: Array) -> Array:
    """Compose the 3x4 projection P = K [R | t]."""
    K = jnp.asarray(K)
    R = jnp.asarray(R)
    t = jnp.asarray(t).reshape(3)
    rt = jnp.concatenate([R, t[:, None]], axis=1)
    return K @ rt


def _camera_points(points: Array, R: Array | None, t: Array | None) -> Array:
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if R is None:
        R = jnp.eye(3, dtype=points.dtype)
    else:
        R = jnp.asarray(R)
    if t is None:
        t = jnp.zeros(3, dtype=points.dtype)
    else:
        t = jnp.asarray(t).reshape(3)
    # p^c = R p^l + t, batched over N.
    return points @ R.T + t


def project_pinhole(
    points: Array,
    K: Array,
    R: Array | None = None,
    t: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Project N points with a pinhole camera (eq. 7).

    Args:
        points: LiDAR-frame points, shape (N, 3).
        K: 3x3 intrinsics K^v.
        R: 3x3 rotation R^v. Identity if omitted (points already in camera frame).
        t: translation t^v, shape (3,). Zero if omitted.

    Returns:
        uv: pixel coordinates (u^v, v^v), shape (N, 2). Invalid depths are 0.
        z_c: camera-frame depth z^c, shape (N,).
        valid: True where z^c > 0 (in front of the camera), shape (N,).
    """
    K = jnp.asarray(K)
    p_c = _camera_points(points, R, t)
    z_c = p_c[:, 2]
    valid = z_c > 0
    # Avoid div-by-zero for points on or behind the image plane.
    safe_z = jnp.where(valid, z_c, jnp.ones_like(z_c))
    homog = p_c @ K.T
    uv = homog[:, :2] / safe_z[:, None]
    uv = jnp.where(valid[:, None], uv, jnp.zeros_like(uv))
    return uv, z_c, valid


def project_P(points: Array, P: Array) -> tuple[Array, Array, Array]:
    """Project N points with a 3x4 matrix P = K [R | t] (eq. 7 equivalent).

    For a standard pinhole K the third homogeneous coordinate equals z^c.
    """
    points = jnp.asarray(points)
    P = jnp.asarray(P)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    ones = jnp.ones((points.shape[0], 1), dtype=points.dtype)
    homog_world = jnp.concatenate([points, ones], axis=1)
    homog = homog_world @ P.T
    z_c = homog[:, 2]
    valid = z_c > 0
    safe_z = jnp.where(valid, z_c, jnp.ones_like(z_c))
    uv = homog[:, :2] / safe_z[:, None]
    uv = jnp.where(valid[:, None], uv, jnp.zeros_like(uv))
    return uv, z_c, valid
