"""Branched flow matching (Sec. 5.2) and the CNF action (Eq. 9).

Interpolant along a discrete trajectory (Sec. 5.2):

    k = clip(floor(t / dt), 0, T-1),   dt = 1/T
    beta = (t - k dt) / dt in [0, 1]
    x_t^i = (1-beta) x_k^{*,i} + beta x_{k+1}^{*,i}
    u_t(x_t^i) = (1-beta) V_{k,i}^* + beta V_{k',i}^*
    k' = min(k+1, T-1)

At t=0 this recovers the source positions and V_0; at t=1 the target
positions x_T and the last discrete velocity V_{T-1}.

Branched flow-matching loss:

    L_BFM(theta) = E_{i ~ Unif[1,N], t ~ Unif[0,1]}
                     || v_theta(x_t^i, t) - u_t(x_t^i) ||^2

Optional CNF form (Eq. 9):

    L^alpha(theta) = int_0^1 int ||v_theta|| p_theta^alpha dx dt
                     - lambda E[log p_theta(x_{t1}, t1)]
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import random


def _time_index(t: jax.Array, n_steps: int) -> tuple[jax.Array, jax.Array]:
    """k = clip(floor(t * T), 0, T-1) and beta in [0, 1]."""
    t = jnp.asarray(t)
    scaled = t * n_steps
    k = jnp.floor(scaled).astype(jnp.int32)
    k = jnp.clip(k, 0, n_steps - 1)
    beta = scaled - k.astype(t.dtype)
    beta = jnp.clip(beta, 0.0, 1.0)
    return k, beta


def path_interpolant(
    positions: jax.Array,
    velocities: jax.Array,
    t: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Linear interpolant of all particles at scalar time t.

    Args:
        positions: (T+1, N, D)
        velocities: (T, N, D)
        t: scalar in [0, 1]

    Returns:
        x_t (N, D), u_t (N, D)
    """
    n_steps = velocities.shape[0]
    k, beta = _time_index(t, n_steps)
    x_k = positions[k]
    x_k1 = positions[k + 1]
    v_k = velocities[k]
    v_k1 = velocities[jnp.minimum(k + 1, n_steps - 1)]
    xt = (1.0 - beta) * x_k + beta * x_k1
    ut = (1.0 - beta) * v_k + beta * v_k1
    return xt, ut


def sample_interpolant(
    positions: jax.Array,
    velocities: jax.Array,
    t: jax.Array,
    particle_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Interpolant at batched times and particle indices (Sec. 5.2).

    Args:
        positions: (T+1, N, D)
        velocities: (T, N, D)
        t: (B,) times in [0, 1]
        particle_index: (B,) integer indices in [0, N)

    Returns:
        x_t (B, D), u_t (B, D)
    """
    n_steps = velocities.shape[0]
    k, beta = _time_index(t, n_steps)
    k1 = k + 1
    kv = jnp.minimum(k + 1, n_steps - 1)
    x_k = positions[k, particle_index]
    x_k1 = positions[k1, particle_index]
    v_k = velocities[k, particle_index]
    v_k1 = velocities[kv, particle_index]
    b = beta[:, None]
    xt = (1.0 - b) * x_k + b * x_k1
    ut = (1.0 - b) * v_k + b * v_k1
    return xt, ut


def bfm_loss_from_vectors(v_pred: jax.Array, u_target: jax.Array) -> jax.Array:
    """L_BFM = mean ||v_theta(x_t, t) - u_t(x_t)||^2."""
    err = v_pred - u_target
    return jnp.mean(jnp.sum(err * err, axis=-1))


def bfm_loss(
    apply_fn: Callable[[jax.Array, jax.Array], jax.Array],
    positions: jax.Array,
    velocities: jax.Array,
    t: jax.Array,
    particle_index: jax.Array,
) -> jax.Array:
    """Evaluate L_BFM on a mini-batch of (t, i) pairs."""
    xt, ut = sample_interpolant(positions, velocities, t, particle_index)
    pred = apply_fn(xt, t)
    return bfm_loss_from_vectors(pred, ut)


def sample_bfm_batch(
    key: jax.Array,
    n_particles: int,
    batch: int,
) -> tuple[jax.Array, jax.Array]:
    """Draw t ~ Unif[0,1] and i ~ Unif{0,...,N-1} as in Sec. 5.2."""
    key_t, key_i = random.split(key)
    t = random.uniform(key_t, (batch,))
    idx = random.randint(key_i, (batch,), 0, n_particles)
    return t, idx


def mlp_init(
    key: jax.Array,
    in_dim: int,
    hidden: int,
    out_dim: int,
    n_hidden: int = 2,
) -> list[tuple[jax.Array, jax.Array]]:
    """Tiny SiLU MLP. Paper uses 64-d time embedding and 3x256; we do not."""
    keys = random.split(key, n_hidden + 1)
    dims = [in_dim] + [hidden] * n_hidden + [out_dim]
    params: list[tuple[jax.Array, jax.Array]] = []
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        w = random.normal(keys[i], (din, dout)) * jnp.sqrt(1.0 / din)
        b = jnp.zeros((dout,))
        params.append((w, b))
    return params


def mlp_apply(
    params: Sequence[tuple[jax.Array, jax.Array]],
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """v_theta(x, t) with time concatenated onto x."""
    t = jnp.asarray(t, dtype=x.dtype)
    if t.ndim == x.ndim:
        t_feat = t
    elif t.ndim == x.ndim - 1:
        t_feat = t[..., None]
    else:
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
    h = jnp.concatenate([x, t_feat], axis=-1)
    for w, b in params[:-1]:
        h = jax.nn.silu(h @ w + b)
    w, b = params[-1]
    return h @ w + b


def cnf_branched_integrand(
    velocity: jax.Array,
    density: jax.Array,
    alpha: float,
) -> jax.Array:
    """Pointwise ||v|| p^alpha from Eq. 9."""
    speed = jnp.linalg.norm(velocity, axis=-1)
    return speed * (density**alpha)


def cnf_objective(
    velocity: jax.Array,
    log_density: jax.Array,
    alpha: float,
    lam: float = 1.0,
    log_p_terminal: jax.Array | None = None,
) -> jax.Array:
    """Monte Carlo stand-in for Eq. 9 given samples of p_theta.

    int ||v|| p^alpha dx = E_{x ~ p}[ ||v|| p^{alpha-1} ].
    The time integral is left to the caller (average over t).
    """
    density = jnp.exp(log_density)
    # Samples of p: int ||v|| p^alpha dx = E_{x ~ p}[ ||v|| p^{alpha-1} ].
    speed = jnp.linalg.norm(velocity, axis=-1)
    action = jnp.mean(speed * (density ** (alpha - 1.0)))
    if log_p_terminal is None:
        return action
    return action - lam * jnp.mean(log_p_terminal)
