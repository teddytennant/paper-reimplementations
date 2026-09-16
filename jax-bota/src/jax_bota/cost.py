"""Branched transport costs from arXiv:2609.15072.

Paper forms implemented here
----------------------------
Atomic Benamou-Brenier branched action (Eq. 7):

    B_alpha = inf_{v, rho} int_0^1 sum_{i in I_t} |v_{t,i}| (rho_{t,i})^alpha dt

with instantaneous atomic cost (main text after Eq. 7 / appendix Lemma 1)

    F(rho_t) = sum_i ||v_{t,i}|| rho_{t,i}^alpha.

Soft-atomic cost C_soft (Lemma 1, used by the discrete solver in Sec. 5.1):

    w_jk = softmax_k( -||x_j - c_k||^2 / (2 sigma^2) )
    m_k  = sum_j w_jk rho_j          (main text: rho_j = 1/B)
    |v_k| = (sum_j w_jk |v_j|) / (sum_j w_jk)     # default, main text
    C_soft = sum_k |v_k| m_k^alpha

The appendix variant replaces |v_k| by the Euclidean norm of the weighted
mean velocity. Both are available via ``speed_mode``.

Static edge cost sum_e length_e * mass_e^alpha is the unit-time special case
of the atomic action on a piecewise-linear branched graph (Xia). That is
what the 2-source / 2-sink topology comparison evaluates.

Defaults not stated in the paper: sigma=0.5, epsilon=1e-8, axis-aligned
bounding-box proxy grid. See README.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

SpeedMode = Literal["speed", "vector"]


def assignment_weights(
    positions: jax.Array,
    centers: jax.Array,
    sigma: float,
) -> jax.Array:
    """Softmax assignment w_jk of particles j to proxy atoms k (Lemma 1).

    Args:
        positions: (N, D)
        centers: (K, D)
        sigma: Gaussian kernel bandwidth.

    Returns:
        weights of shape (N, K), rows sum to 1.
    """
    diff = positions[:, None, :] - centers[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    logits = -dist_sq / (2.0 * (sigma * sigma))
    return jax.nn.softmax(logits, axis=-1)


def atomic_branched_cost(
    velocities: jax.Array,
    masses: jax.Array,
    alpha: float,
) -> jax.Array:
    """Instantaneous atomic cost F = sum_i ||v_i|| m_i^alpha (Eq. 7 integrand).

    Args:
        velocities: (N, D)
        masses: (N,) non-negative masses (probability or unnormalized)
        alpha: branch exponent in (0, 1]; alpha=1 is linear in mass.
    """
    speed = jnp.linalg.norm(velocities, axis=-1)
    return jnp.sum(speed * (masses**alpha))


def edge_transport_cost(
    lengths: jax.Array,
    masses: jax.Array,
    alpha: float,
) -> jax.Array:
    """Static BOT edge cost sum_e ell_e * m_e^alpha.

    Equivalent to integrating |v| m^alpha dt along a constant-speed edge
    that covers length ell in unit time (|v|=ell).
    """
    return jnp.sum(lengths * (masses**alpha))


def soft_atomic_cost(
    positions: jax.Array,
    velocities: jax.Array,
    centers: jax.Array,
    alpha: float,
    sigma: float = 0.5,
    masses: jax.Array | None = None,
    speed_mode: SpeedMode = "speed",
    eps: float = 1e-8,
) -> jax.Array:
    """C_soft of Lemma 1 / discrete solver (Sec. 5.1).

    Args:
        positions: (N, D) particle locations x_j
        velocities: (N, D) particle velocities v_j
        centers: (K, D) proxy atom locations c_k
        alpha: branch exponent
        sigma: kernel bandwidth (paper does not specify a value)
        masses: (N,) particle masses rho_j; default 1/N as in the main text
        speed_mode: ``speed`` averages |v_j| (main text Lemma 1);
            ``vector`` uses ||sum w_jk v_j / sum w_jk|| (appendix)
        eps: denominator floor
    """
    n = positions.shape[0]
    if masses is None:
        masses = jnp.full((n,), 1.0 / n, dtype=positions.dtype)
    weights = assignment_weights(positions, centers, sigma)
    soft_mass = masses @ weights
    col_mass = jnp.sum(weights, axis=0) + eps
    if speed_mode == "speed":
        speed = jnp.linalg.norm(velocities, axis=-1)
        v_bar = (speed @ weights) / col_mass
    elif speed_mode == "vector":
        v_mean = (weights.T @ velocities) / col_mass[:, None]
        v_bar = jnp.linalg.norm(v_mean, axis=-1)
    else:
        raise ValueError(f"unknown speed_mode {speed_mode!r}")
    return jnp.sum(v_bar * (soft_mass**alpha))


def integrated_soft_cost(
    positions: jax.Array,
    velocities: jax.Array,
    centers: jax.Array,
    alpha: float,
    dt: float,
    sigma: float = 0.5,
    masses: jax.Array | None = None,
    speed_mode: SpeedMode = "speed",
) -> jax.Array:
    """Riemann sum sum_k C_soft(X_k, V_k) Delta t (discrete analogue of Eq. 7).

    Args:
        positions: (T+1, N, D)
        velocities: (T, N, D)
        dt: 1/T
    """

    def body(k, acc):
        return acc + soft_atomic_cost(
            positions[k],
            velocities[k],
            centers,
            alpha,
            sigma=sigma,
            masses=masses,
            speed_mode=speed_mode,
        ) * dt

    t_steps = velocities.shape[0]
    return jax.lax.fori_loop(0, t_steps, body, jnp.asarray(0.0, dtype=positions.dtype))


def bounding_grid_centers(
    points: jax.Array,
    n_per_axis: int = 4,
    pad: float = 0.25,
) -> jax.Array:
    """Axis-aligned grid covering the bounding box of ``points``.

    The paper does not specify how proxy centers {c_k} are placed. This kernel
    uses a regular grid on the padded bounding box of source and target.
    """
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    span = hi - lo
    span = jnp.where(span < 1e-6, jnp.ones_like(span), span)
    lo = lo - pad * span
    hi = hi + pad * span
    dim = points.shape[-1]
    if dim == 1:
        return jnp.linspace(lo[0], hi[0], n_per_axis)[:, None]
    if dim == 2:
        xs = jnp.linspace(lo[0], hi[0], n_per_axis)
        ys = jnp.linspace(lo[1], hi[1], n_per_axis)
        xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
        return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    axes = [jnp.linspace(lo[i], hi[i], n_per_axis) for i in range(dim)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=-1)


# ---------------------------------------------------------------------------
# Hand-built 2-source / 2-sink topologies (no training).
# Sources A=(-0.1, 0), B=(0.1, 0); sinks C=(-0.1, 2), D=(0.1, 2).
# Independent: A->C and B->D (length 2, mass 1 each).
# Branched: merge at J1=(0, 0.2), shared trunk to J2=(0, 1.8), then split.
# ---------------------------------------------------------------------------

_LEG = (0.1**2 + 0.2**2) ** 0.5
_TRUNK = 1.6
_STRAIGHT = 2.0

INDEPENDENT_LENGTHS = jnp.asarray([_STRAIGHT, _STRAIGHT])
INDEPENDENT_MASSES = jnp.asarray([1.0, 1.0])
BRANCHED_LENGTHS = jnp.asarray([_LEG, _LEG, _TRUNK, _LEG, _LEG])
BRANCHED_MASSES = jnp.asarray([1.0, 1.0, 2.0, 1.0, 1.0])


def two_source_two_sink_costs(alpha: float) -> tuple[jax.Array, jax.Array]:
    """Evaluate static BOT cost of the two hand topologies at a given alpha.

    Returns (independent_cost, branched_cost).
    """
    independent = edge_transport_cost(INDEPENDENT_LENGTHS, INDEPENDENT_MASSES, alpha)
    branched = edge_transport_cost(BRANCHED_LENGTHS, BRANCHED_MASSES, alpha)
    return independent, branched
