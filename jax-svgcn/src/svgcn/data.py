"""Synthetic 16-joint 3D gait sequences with 4 emotion classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from svgcn.skeleton import NUM_JOINTS

NUM_CLASSES = 4

# Neutral standing pose, y-up, x right, z forward.
_TPOSE = jnp.asarray(
    [
        [0.00, 0.00, 0.00],  # root
        [0.00, 0.30, 0.00],  # spine
        [0.00, 0.55, 0.00],  # neck
        [0.00, 0.75, 0.00],  # head
        [-0.20, 0.52, 0.00],  # l shoulder
        [-0.38, 0.28, 0.00],  # l elbow
        [-0.42, 0.04, 0.00],  # l wrist
        [0.20, 0.52, 0.00],  # r shoulder
        [0.38, 0.28, 0.00],  # r elbow
        [0.42, 0.04, 0.00],  # r wrist
        [-0.10, -0.02, 0.00],  # l hip
        [-0.12, -0.40, 0.00],  # l knee
        [-0.12, -0.78, 0.00],  # l ankle
        [0.10, -0.02, 0.00],  # r hip
        [0.12, -0.40, 0.00],  # r knee
        [0.12, -0.78, 0.00],  # r ankle
    ],
    dtype=jnp.float32,
)


def tpose() -> jnp.ndarray:
    return _TPOSE


def generate_synthetic(
    key: jax.Array,
    batch_size: int = 64,
    t_max: int = 32,
    min_len: int = 12,
    num_classes: int = NUM_CLASSES,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Class-dependent pose / swing so a tiny network can fit.

    Returns ``coords (N, T, J, 3)``, ``lengths (N,)``, ``labels (N,)``.
    Frames after ``lengths`` are zeros.
    """
    k_len, k_lab, k_noise, k_phase = jax.random.split(key, 4)
    labels = jax.random.randint(k_lab, (batch_size,), 0, num_classes)
    lengths = jax.random.randint(k_len, (batch_size,), min_len, t_max + 1)
    t = jnp.arange(t_max, dtype=jnp.float32)[None, :]
    phase = jax.random.uniform(k_phase, (batch_size, 1), minval=0.0, maxval=2 * jnp.pi)
    # Class cues: static pose bias (easy for GAP) plus gait swing.
    # 0 happy, 1 sad, 2 angry, 3 neutral
    arm_amp = jnp.take(jnp.array([0.28, 0.03, 0.20, 0.10], dtype=jnp.float32), labels)
    stride = jnp.take(jnp.array([0.16, 0.04, 0.22, 0.09], dtype=jnp.float32), labels)
    slump = jnp.take(jnp.array([-0.04, 0.14, -0.08, 0.02], dtype=jnp.float32), labels)
    # Lateral lean / arm raise offsets that survive averaging over time.
    lean_x = jnp.take(jnp.array([0.00, -0.06, 0.08, 0.00], dtype=jnp.float32), labels)
    raise_y = jnp.take(jnp.array([0.12, -0.10, 0.04, 0.00], dtype=jnp.float32), labels)
    fwd_z = jnp.take(jnp.array([0.08, -0.06, 0.12, 0.00], dtype=jnp.float32), labels)
    freq = jnp.take(jnp.array([1.2, 0.7, 1.6, 1.0], dtype=jnp.float32), labels)
    osc = jnp.sin(2 * jnp.pi * freq[:, None] * t / t_max + phase)

    pose = jnp.broadcast_to(_TPOSE[None, None, :, :], (batch_size, t_max, NUM_JOINTS, 3))
    bias = jnp.zeros((batch_size, t_max, NUM_JOINTS, 3), dtype=jnp.float32)
    slump_y = -slump[:, None]
    lean = lean_x[:, None]
    raise_ = raise_y[:, None]
    fwd = fwd_z[:, None]
    bias = bias.at[:, :, 1, 1].set(slump_y)
    bias = bias.at[:, :, 2, 1].set(1.5 * slump_y)
    bias = bias.at[:, :, 3, 1].set(2.0 * slump_y)
    bias = bias.at[:, :, 3, 0].set(lean)
    bias = bias.at[:, :, 6, 1].set(raise_)
    bias = bias.at[:, :, 9, 1].set(raise_)
    bias = bias.at[:, :, 6, 2].set(fwd)
    bias = bias.at[:, :, 9, 2].set(-fwd)

    swing = jnp.zeros_like(pose)
    a = (arm_amp[:, None] * osc)[:, :, None]
    s = (stride[:, None] * osc)[:, :, None]
    swing = swing.at[:, :, 6, 2].add(a[:, :, 0])
    swing = swing.at[:, :, 5, 2].set(0.5 * a[:, :, 0])
    swing = swing.at[:, :, 9, 2].add(-a[:, :, 0])
    swing = swing.at[:, :, 8, 2].set(-0.5 * a[:, :, 0])
    swing = swing.at[:, :, 12, 2].set(-s[:, :, 0])
    swing = swing.at[:, :, 11, 2].set(-0.5 * s[:, :, 0])
    swing = swing.at[:, :, 15, 2].set(s[:, :, 0])
    swing = swing.at[:, :, 14, 2].set(0.5 * s[:, :, 0])

    noise = 0.005 * jax.random.normal(k_noise, pose.shape, dtype=jnp.float32)
    coords = pose + bias + swing + noise
    mask_t = (jnp.arange(t_max)[None, :] < lengths[:, None])[:, :, None, None]
    coords = coords * mask_t
    return coords, lengths.astype(jnp.int32), labels.astype(jnp.int32)
