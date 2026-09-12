"""16-joint kinematic tree (Kinect / E-Gait style) and spatial adjacency."""

from __future__ import annotations

import jax.numpy as jnp

NUM_JOINTS = 16

# Joint 0 is the root. parent[j] = -1 for the root.
PARENTS: tuple[int, ...] = (
    -1,  # 0 root
    0,  # 1 spine
    1,  # 2 neck
    2,  # 3 head
    2,  # 4 left shoulder
    4,  # 5 left elbow
    5,  # 6 left wrist
    2,  # 7 right shoulder
    7,  # 8 right elbow
    8,  # 9 right wrist
    0,  # 10 left hip
    10,  # 11 left knee
    11,  # 12 left ankle
    0,  # 13 right hip
    13,  # 14 right knee
    14,  # 15 right ankle
)

JOINT_NAMES: tuple[str, ...] = (
    "root",
    "spine",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)

# (J1, J2, J3) with J2 the middle joint that receives f_theta (eq. 1).
# Limb joints: parent–joint–child. Endpoints: parent–joint–root.
# Root: left-hip–root–right-hip.
ANGLE_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (10, 0, 13),  # root
    (0, 1, 2),  # spine
    (1, 2, 3),  # neck
    (2, 3, 0),  # head (parent–head–root)
    (2, 4, 5),  # left shoulder
    (4, 5, 6),  # left elbow
    (5, 6, 0),  # left wrist
    (2, 7, 8),  # right shoulder
    (7, 8, 9),  # right elbow
    (8, 9, 0),  # right wrist
    (0, 10, 11),  # left hip
    (10, 11, 12),  # left knee
    (11, 12, 0),  # left ankle
    (0, 13, 14),  # right hip
    (13, 14, 15),  # right knee
    (14, 15, 0),  # right ankle
)


def _adjacency() -> jnp.ndarray:
    a = jnp.eye(NUM_JOINTS, dtype=jnp.float32)
    for child, parent in enumerate(PARENTS):
        if parent >= 0:
            a = a.at[child, parent].set(1.0)
            a = a.at[parent, child].set(1.0)
    return a


ADJACENCY = _adjacency()


def normalized_adjacency(adj: jnp.ndarray | None = None) -> jnp.ndarray:
    r"""Symmetric degree-normalized adjacency, D^{-1/2} A D^{-1/2}.

    ``ADJACENCY`` already includes the identity (self-loops).
    """
    if adj is None:
        adj = ADJACENCY
    deg = jnp.sum(adj, axis=1)
    inv_sqrt = jnp.where(deg > 0, 1.0 / jnp.sqrt(deg), 0.0)
    d = jnp.diag(inv_sqrt)
    return d @ adj @ d
