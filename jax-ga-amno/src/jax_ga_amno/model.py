"""Tiny stacked GA-AMNO kernel: allocation + L gauge-aware layers.

Table 2 steps 1–8 only. Steps 9–12 (importance-modulated readout, coarse
upsampling, spectral residual, differential correction) are not implemented.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_ga_amno.allocation import (
    allocate,
    direction_from_logits,
    importance_from_logits,
    physical_indicators,
)
from jax_ga_amno.geometry import (
    edge_descriptors,
    four_connected_neighbors,
    local_jacobians,
    neighbor_mask,
    reference_grid,
)
from jax_ga_amno.transport import gauge_layer

Array = jax.Array
Params = dict


class GAAMNOConfig(NamedTuple):
    height: int
    width: int
    hidden: int = 8
    rank: int = 2
    n_layers: int = 2
    in_channels: int = 1
    out_channels: int = 1
    i_min: float = 0.05
    s_i: float = 1.0
    delta_max: float = 0.08
    eps: float = 1e-8
    omega_min: float = 0.0
    omega_max: float = 1.0
    skip_transport: bool = False


def _linear(key: Array, din: int, dout: int, scale: float | None = None) -> Params:
    k_w, k_b = jax.random.split(key)
    if scale is None:
        scale = (1.0 / din) ** 0.5
    w = scale * jax.random.normal(k_w, (din, dout))
    b = jnp.zeros((dout,))
    return {"W": w, "b": b}


def _mlp(params: list[Params], x: Array) -> Array:
    for layer in params[:-1]:
        x = jax.nn.gelu(x @ layer["W"] + layer["b"])
    last = params[-1]
    return x @ last["W"] + last["b"]


def _init_mlp(key: Array, sizes: tuple[int, ...]) -> list[Params]:
    keys = jax.random.split(key, len(sizes) - 1)
    return [_linear(k, din, dout) for k, din, dout in zip(keys, sizes[:-1], sizes[1:])]


def _init_heads(key: Array, edge_dim: int, hidden: int, rank: int) -> Params:
    keys = jax.random.split(key, 6)
    s = 0.1 / (edge_dim ** 0.5)
    params = {
        "W_d": s * jax.random.normal(keys[0], (edge_dim, hidden)),
        "b_d": jnp.ones((hidden,)),
        "W_U": s * jax.random.normal(keys[1], (edge_dim, hidden * rank)),
        "b_U": jnp.zeros((hidden * rank,)),
        "W_V": s * jax.random.normal(keys[2], (edge_dim, hidden * rank)),
        "b_V": jnp.zeros((hidden * rank,)),
        "W_gamma": s * jax.random.normal(keys[3], (edge_dim, 1)),
        "b_gamma": jnp.array([-2.0]),
        "W_a": s * jax.random.normal(keys[4], (edge_dim, 1)),
        "b_a": jnp.zeros((1,)),
        "rho": {
            "W": jnp.eye(hidden) + 0.01 * jax.random.normal(keys[5], (hidden, hidden)),
            "b": jnp.zeros((hidden,)),
        },
    }
    return params


def init_ga_amno(key: Array, config: GAAMNOConfig) -> Params:
    """Initialise encoder, η_θ, ψ_θ, L transport layers, and a linear readout."""
    dim = 2
    n_ind = 2
    hidden = config.hidden
    edge_dim = dim + dim * dim + dim * dim + hidden + hidden
    k_enc, k_eta, k_psi, k_read, k_layers = jax.random.split(key, 5)
    layer_keys = jax.random.split(k_layers, config.n_layers)
    return {
        "encoder": _init_mlp(k_enc, (config.in_channels, hidden, hidden)),
        "eta": _init_mlp(k_eta, (hidden + n_ind, hidden, 1)),
        "psi": _init_mlp(k_psi, (hidden, hidden, dim)),
        "layers": [_init_heads(k, edge_dim, hidden, config.rank) for k in layer_keys],
        "readout": _linear(k_read, hidden, config.out_channels),
    }


def ga_amno_forward(params: Params, field: Array, config: GAAMNOConfig) -> dict[str, Array]:
    """Allocation + stacked gauge layers on a regular 2D grid field.

    ``field``: (H, W) or (H, W, C). Coordinates live in [omega_min, omega_max]^2.
    """
    if field.ndim == 2:
        field = field[..., None]
    height, width, _ = field.shape
    n = height * width
    x_r = reference_grid(height, width, config.omega_min, config.omega_max)
    neighbors = four_connected_neighbors(height, width)
    mask = neighbor_mask(neighbors)
    q = physical_indicators(field).reshape(n, -1)
    v = field.reshape(n, -1)
    p = _mlp(params["encoder"], v)
    importance = importance_from_logits(_mlp(params["eta"], jnp.concatenate([p, q], axis=-1)))
    importance = importance.reshape(n)
    direction = direction_from_logits(_mlp(params["psi"], p))
    x_a, w, dx = allocate(
        x_r,
        importance,
        direction,
        i_min=config.i_min,
        s_i=config.s_i,
        delta_max=config.delta_max,
        eps=config.eps,
        omega_min=config.omega_min,
        omega_max=config.omega_max,
    )
    jacobians = local_jacobians(x_a, x_r, neighbors)
    edges = edge_descriptors(x_a, jacobians, p, neighbors)
    h = p
    m = jnp.zeros_like(h)
    for layer in params["layers"]:
        h, m, _alpha = gauge_layer(
            layer,
            h,
            edges,
            neighbors,
            skip_transport=config.skip_transport,
            mask=mask,
        )
    out = h @ params["readout"]["W"] + params["readout"]["b"]
    return {
        "h": h,
        "m": m,
        "out": out.reshape(height, width, config.out_channels),
        "x_a": x_a,
        "x_r": x_r,
        "w": w,
        "dx": dx,
        "I": importance,
        "J": jacobians,
        "o": direction,
    }
