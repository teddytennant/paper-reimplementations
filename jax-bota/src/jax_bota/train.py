"""Two-stage BOTA training loop: discrete BOT then branched flow matching."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from jax_bota.flow import mlp_apply, mlp_init, sample_bfm_batch, sample_interpolant
from jax_bota.particles import DiscreteSolve, solve_discrete


class AdamState(NamedTuple):
    m: object
    v: object
    t: jax.Array


class TrainResult(NamedTuple):
    params: object
    losses: jax.Array
    discrete: DiscreteSolve


def sample_gmm_pair(
    key: jax.Array,
    n: int,
    dim: int = 2,
    std: float = 0.12,
) -> tuple[jax.Array, jax.Array]:
    """2 Gaussians at y=-1 to 2 Gaussians at y=+1 (left/right blobs).

    Particles are sorted by the first coordinate so index pairing in the
    discrete solver maps left-to-left and right-to-right.
    """
    k0, k1, k2, k3 = random.split(key, 4)
    lab0 = random.bernoulli(k0, 0.5, (n,))
    x0 = random.normal(k1, (n, dim)) * std
    x0 = x0.at[:, 0].add(jnp.where(lab0, -1.0, 1.0))
    if dim > 1:
        x0 = x0.at[:, 1].add(-1.0)
    lab1 = random.bernoulli(k2, 0.5, (n,))
    x1 = random.normal(k3, (n, dim)) * std
    x1 = x1.at[:, 0].add(jnp.where(lab1, -1.0, 1.0))
    if dim > 1:
        x1 = x1.at[:, 1].add(1.0)
    x0 = x0[jnp.argsort(x0[:, 0])]
    x1 = x1[jnp.argsort(x1[:, 0])]
    return x0, x1


def adam_init(params) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(m=zeros, v=zeros, t=jnp.array(0, dtype=jnp.int32))


def adam_update(
    params,
    grads,
    state: AdamState,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[object, AdamState]:
    t = state.t + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state.m, grads)
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1.0 - b2) * (g * g), state.v, grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1 ** t.astype(jnp.float32)), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2 ** t.astype(jnp.float32)), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, AdamState(m=m, v=v, t=t)


def train_bfm(
    positions: jax.Array,
    velocities: jax.Array,
    n_steps: int = 200,
    hidden: int = 32,
    n_hidden: int = 2,
    lr: float = 3e-3,
    batch: int = 64,
    seed: int = 0,
) -> tuple[object, jax.Array]:
    """Fit a tiny MLP to the discrete interpolant velocity (Stage B)."""
    n_particles, dim = positions.shape[1], positions.shape[2]
    key = random.PRNGKey(seed)
    key, k_init = random.split(key)
    params = mlp_init(k_init, in_dim=dim + 1, hidden=hidden, out_dim=dim, n_hidden=n_hidden)
    state = adam_init(params)

    def loss_fn(p, t, idx):
        xt, ut = sample_interpolant(positions, velocities, t, idx)
        pred = mlp_apply(p, xt, t)
        err = pred - ut
        return jnp.mean(jnp.sum(err * err, axis=-1))

    @jax.jit
    def step(params, state, key):
        t, idx = sample_bfm_batch(key, n_particles, batch)
        loss, grads = jax.value_and_grad(loss_fn)(params, t, idx)
        params, state = adam_update(params, grads, state, lr=lr)
        return params, state, loss

    losses = []
    for _ in range(n_steps):
        key, k = random.split(key)
        params, state, loss = step(params, state, k)
        losses.append(loss)
    return params, jnp.stack(losses)


def train_bota(
    x0: jax.Array,
    x_target: jax.Array,
    alpha: float = 0.5,
    n_time: int = 6,
    n_discrete_opt: int = 25,
    n_bfm_steps: int = 200,
    hidden: int = 32,
    lr_discrete: float = 0.08,
    lr_bfm: float = 3e-3,
    seed: int = 0,
    **discrete_kwargs,
) -> TrainResult:
    """Stage A discrete BOT, then Stage B branched flow matching."""
    discrete = solve_discrete(
        x0,
        x_target,
        alpha=alpha,
        n_steps=n_time,
        n_opt=n_discrete_opt,
        lr=lr_discrete,
        **discrete_kwargs,
    )
    params, losses = train_bfm(
        discrete.positions,
        discrete.velocities,
        n_steps=n_bfm_steps,
        hidden=hidden,
        lr=lr_bfm,
        seed=seed,
    )
    return TrainResult(params=params, losses=losses, discrete=discrete)
