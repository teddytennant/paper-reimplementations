"""Shared-layer DEQ map F_θ(z, x) and residual r = F(z, x) - z (eq 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def residual(F, z, x):
    """r = F_θ(z, x) - z."""
    return F(z, x) - z


def init_shared_params(key, hidden=16, input_dim=24, mlp_hidden=16):
    """Tiny linear path plus residual MLP. Hidden ~16 for fast tests."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    Wz = 0.15 * jax.random.normal(k1, (hidden, hidden)) / jnp.sqrt(hidden)
    Wx = 0.15 * jax.random.normal(k2, (input_dim, hidden)) / jnp.sqrt(input_dim)
    W1 = jax.random.normal(k3, (hidden + input_dim, mlp_hidden)) / jnp.sqrt(
        hidden + input_dim
    )
    W2 = 0.1 * jax.random.normal(k4, (mlp_hidden, hidden)) / jnp.sqrt(mlp_hidden)
    return {
        "Wz": Wz,
        "Wx": Wx,
        "b": jnp.zeros((hidden,)),
        "W1": W1,
        "b1": jnp.zeros((mlp_hidden,)),
        "W2": W2,
        "b2": jnp.zeros((hidden,)),
    }


def shared_layer(params, z, x):
    """F_θ(z, x) -> z. Linear map of (z, x) plus a small residual MLP."""
    lin = z @ params["Wz"] + x @ params["Wx"] + params["b"]
    h = jnp.concatenate([z, x], axis=-1)
    h = jnp.tanh(h @ params["W1"] + params["b1"])
    res = h @ params["W2"] + params["b2"]
    return jnp.tanh(lin + res)


def apply_F(params, z, x):
    return shared_layer(params, z, x)


def init_head(key, hidden, out_dim):
    W = jax.random.normal(key, (hidden, out_dim)) / jnp.sqrt(hidden)
    return {"W": W, "b": jnp.zeros((out_dim,))}


def head(params, z):
    return z @ params["W"] + params["b"]


def count_params(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return int(sum(int(jnp.size(p)) for p in leaves))


def init_stacked_params(key, n_layers, **kwargs):
    keys = jax.random.split(key, n_layers)
    return [init_shared_params(k, **kwargs) for k in keys]


def stacked_apply(param_list, z, x):
    for p in param_list:
        z = shared_layer(p, z, x)
    return z
