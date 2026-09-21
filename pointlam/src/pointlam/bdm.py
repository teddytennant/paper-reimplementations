"""Simplified Bi-Directional Mamba (BDM).

Paper: axis-wise scans (X then Y) or Hilbert serialization, full Mamba.
Here: for X then Y, sort points by that coordinate, run a tiny input-gated
selective SSM forward and backward, add the two directions, residual back
onto the unsorted features. State width is 8. No Hilbert curve.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ScanParams(NamedTuple):
    A_log: jax.Array
    W_dt: jax.Array
    b_dt: jax.Array
    W_B: jax.Array
    W_C: jax.Array
    D_skip: jax.Array


class BDMParams(NamedTuple):
    x_fwd: ScanParams
    x_bwd: ScanParams
    y_fwd: ScanParams
    y_bwd: ScanParams
    W_mix: jax.Array


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def init_scan(key: jax.Array, dim: int, state: int) -> ScanParams:
    k1, k2, k3 = jax.random.split(key, 3)
    a_log = jnp.broadcast_to(
        jnp.log(jnp.arange(1, state + 1, dtype=jnp.float32)), (dim, state)
    )
    return ScanParams(
        A_log=a_log,
        W_dt=jax.random.normal(k1, (dim, dim)) * (0.1 / jnp.sqrt(dim)),
        b_dt=jnp.full((dim,), _inv_softplus(0.05)),
        W_B=jax.random.normal(k2, (dim, state)) * (0.1 / jnp.sqrt(dim)),
        W_C=jax.random.normal(k3, (dim, state)) * (0.1 / jnp.sqrt(dim)),
        D_skip=jnp.ones((dim,)),
    )


def init_bdm(key: jax.Array, dim: int, state: int = 8) -> BDMParams:
    keys = jax.random.split(key, 5)
    return BDMParams(
        x_fwd=init_scan(keys[0], dim, state),
        x_bwd=init_scan(keys[1], dim, state),
        y_fwd=init_scan(keys[2], dim, state),
        y_bwd=init_scan(keys[3], dim, state),
        W_mix=jax.random.normal(keys[4], (dim, dim)) * (0.1 / jnp.sqrt(dim)),
    )


def selective_scan(x: jax.Array, params: ScanParams) -> jax.Array:
    """Input-gated SISO-per-channel selective SSM. x is (L, D)."""
    a = -jnp.exp(params.A_log)
    dt = jax.nn.softplus(x @ params.W_dt + params.b_dt)
    dt = jnp.clip(dt, 1e-4, 0.5)
    b = x @ params.W_B
    c = x @ params.W_C
    dim = x.shape[-1]
    state = params.A_log.shape[-1]
    h0 = jnp.zeros((dim, state), dtype=x.dtype)

    def step(h, inp):
        x_t, dt_t, b_t, c_t = inp
        dA = jnp.exp(dt_t[:, None] * a)
        dB = dt_t[:, None] * b_t[None, :]
        h = dA * h + dB * x_t[:, None]
        y = (h * c_t[None, :]).sum(axis=-1) + params.D_skip * x_t
        return h, y

    _, ys = jax.lax.scan(step, h0, (x, dt, b, c))
    return ys


def _bidirectional(x: jax.Array, fwd: ScanParams, bwd: ScanParams) -> jax.Array:
    yf = selective_scan(x, fwd)
    yb = selective_scan(x[::-1], bwd)[::-1]
    return yf + yb


def _axis_scan(
    points: jax.Array,
    features: jax.Array,
    axis: int,
    fwd: ScanParams,
    bwd: ScanParams,
) -> jax.Array:
    order = jnp.argsort(points[:, axis], stable=True)
    y = _bidirectional(features[order], fwd, bwd)
    return y[jnp.argsort(order, stable=True)]


def apply_bdm(params: BDMParams, points: jax.Array, features: jax.Array) -> jax.Array:
    h = _axis_scan(points, features, 0, params.x_fwd, params.x_bwd)
    h = _axis_scan(points, h, 1, params.y_fwd, params.y_bwd)
    return features + h @ params.W_mix
