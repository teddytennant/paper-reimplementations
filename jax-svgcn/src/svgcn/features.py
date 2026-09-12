r"""Angle (eq. 1) and intra-frame relative motion (eq. 2-3)."""

from __future__ import annotations

import jax.numpy as jnp

from svgcn.skeleton import ANGLE_TRIPLES, NUM_JOINTS

EPS = 1e-8
MOTION_DIM = 10
# 3 coord + 8 linear f_rel + 1 angle + 2 angular f_rel
FEATURE_DIM = 14


def angle_feature(j1: jnp.ndarray, j2: jnp.ndarray, j3: jnp.ndarray) -> jnp.ndarray:
    """Included angle at ``j2`` for triples ``(j1, j2, j3)`` (eq. 1).

    ``j1,j2,j3`` broadcast to shape ``(..., 3)``. Returns radians in ``[0, π]``.
    """
    u = j1 - j2
    v = j3 - j2
    u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    u = u / jnp.maximum(u_norm, EPS)
    v = v / jnp.maximum(v_norm, EPS)
    cos = jnp.sum(u * v, axis=-1)
    cos = jnp.clip(cos, -1.0, 1.0)
    return jnp.arccos(cos)


def angles_from_coords(coords: jnp.ndarray) -> jnp.ndarray:
    """``coords`` is ``(..., T, J, 3)`` or ``(..., J, 3)``. Returns ``(..., T, J)`` / ``(..., J)``."""
    triples = jnp.asarray(ANGLE_TRIPLES, dtype=jnp.int32)
    j1 = coords[..., triples[:, 0], :]
    j2 = coords[..., triples[:, 1], :]
    j3 = coords[..., triples[:, 2], :]
    return angle_feature(j1, j2, j3)


def _time_diff(x: jnp.ndarray, axis: int = -3) -> jnp.ndarray:
    """First difference along time, zero at the first frame."""
    d = jnp.diff(x, axis=axis)
    lead_shape = list(x.shape)
    lead_shape[axis] = 1
    zeros = jnp.zeros(lead_shape, dtype=x.dtype)
    return jnp.concatenate([zeros, d], axis=axis)


def motion_features(coords: jnp.ndarray, angles: jnp.ndarray | None = None) -> jnp.ndarray:
    """Per-joint f_dt in R^10 between adjacent frames.

    Layout: linear vel xyz+mag (4), linear acc xyz+mag (4), angular vel (1),
    angular acc (1). ``coords`` has shape ``(..., T, J, 3)``.
    """
    if angles is None:
        angles = angles_from_coords(coords)
    vel = _time_diff(coords, axis=-3)
    vel_mag = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    acc = _time_diff(vel, axis=-3)
    acc_mag = jnp.linalg.norm(acc, axis=-1, keepdims=True)
    ang_vel = _time_diff(angles, axis=-1)[..., None]
    ang_acc = _time_diff(ang_vel[..., 0], axis=-1)[..., None]
    return jnp.concatenate([vel, vel_mag, acc, acc_mag, ang_vel, ang_acc], axis=-1)


def pm_sign(x: jnp.ndarray) -> jnp.ndarray:
    """Sign in {-1,+1}: zero is mapped to +1 so sigma matches eq. 3."""
    return jnp.where(x >= 0, jnp.ones_like(x), -jnp.ones_like(x))


def relative_sign(f_joint: jnp.ndarray, f_root: jnp.ndarray) -> jnp.ndarray:
    """sigma = sign(f_j) * sign(f_0) (eq. 3)."""
    return pm_sign(f_joint) * pm_sign(f_root)


def relative_motion(
    f_joint: jnp.ndarray,
    f_root: jnp.ndarray,
    eps: float = EPS,
) -> jnp.ndarray:
    """Intra-frame relative motion (eq. 2-3), literal scalar-norm form.

    The ln argument is the ratio of Euclidean norms of the 10-vectors
    (after adding eps to each component). sigma is 10-D, so f_rel shares
    one magnitude and has per-component signs.
    """
    sigma = relative_sign(f_joint, f_root)
    n_j = jnp.linalg.norm(f_joint + eps, axis=-1, keepdims=True)
    n_0 = jnp.linalg.norm(f_root + eps, axis=-1, keepdims=True)
    return sigma * jnp.log(n_j / n_0)


def relative_motion_all(f_delta: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """``f_delta`` shape ``(..., J, 10)`` with root at joint 0."""
    root = f_delta[..., 0, :]
    return relative_motion(f_delta, root[..., None, :], eps=eps)


def build_features(coords: jnp.ndarray, eps: float = EPS) -> jnp.ndarray:
    """Stack 14-D joint features as ``(..., 14, T, J)``.

    Channel layout:
      0:3   coordinates
      3:11  f_rel[0:8] linear motion (paper slice [0:7] read inclusive)
      11    f_theta
      12:14 f_rel[8:10] angular motion (paper slice [8:9] inclusive)
    """
    angles = angles_from_coords(coords)
    f_dt = motion_features(coords, angles)
    f_rel = relative_motion_all(f_dt, eps=eps)
    coord_nctj = jnp.moveaxis(coords, [-3, -2, -1], [-2, -1, -3])
    rel_nctj = jnp.moveaxis(f_rel, [-3, -2, -1], [-2, -1, -3])
    ang_nctj = angles[..., None, :, :]
    linear = rel_nctj[..., 0:8, :, :]
    angular_rel = rel_nctj[..., 8:10, :, :]
    return jnp.concatenate([coord_nctj, linear, ang_nctj, angular_rel], axis=-3)
