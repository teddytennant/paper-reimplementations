"""AdamX moments, cosine controller, and AMSGrad envelope."""

from __future__ import annotations

import jax.numpy as jnp
from jax import tree_util


def _as_vec(tree):
    leaves = tree_util.tree_leaves(tree)
    return jnp.concatenate([jnp.ravel(x) for x in leaves])


def cosine_controller(g, g_prev, lam, delta):
    a = _as_vec(g)
    b = _as_vec(g_prev)
    num = jnp.dot(a, b)
    den = jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), delta)
    c = num / den
    return jnp.exp(lam * c)


def amsgrad_envelope(v_hat, v_tilde_prev):
    return tree_util.tree_map(jnp.maximum, v_tilde_prev, v_hat)


def init(params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, lam=0.1, delta=1e-8):
    zeros = tree_util.tree_map(jnp.zeros_like, params)
    return {
        "count": 0,
        "m": zeros,
        "v": zeros,
        "v_tilde": zeros,
        "g_prev": zeros,
        "lr": lr,
        "b1": b1,
        "b2": b2,
        "eps": eps,
        "lam": lam,
        "delta": delta,
    }


def update(grads, state, params=None):
    del params
    t = state["count"] + 1
    b1 = state["b1"]
    b2 = state["b2"]
    m = tree_util.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads)
    v = tree_util.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * g * g, state["v"], grads)
    m_hat = tree_util.tree_map(lambda mi: mi / (1.0 - b1 ** t), m)
    v_hat = tree_util.tree_map(lambda vi: vi / (1.0 - b2 ** t), v)
    v_tilde = amsgrad_envelope(v_hat, state["v_tilde"])
    gamma = cosine_controller(grads, state["g_prev"], state["lam"], state["delta"])
    if t < 2:
        gamma = 1.0
    lr = state["lr"]
    eps = state["eps"]
    updates = tree_util.tree_map(
        lambda mh, vt: -lr * gamma * mh / (jnp.sqrt(vt) + eps),
        m_hat,
        v_tilde,
    )
    new_state = {
        "count": state["count"] + 1,
        "m": m,
        "v": v,
        "v_tilde": v_tilde,
        "g_prev": grads,
        "lr": state["lr"],
        "b1": state["b1"],
        "b2": state["b2"],
        "eps": state["eps"],
        "lam": state["lam"],
        "delta": state["delta"],
    }
    return updates, new_state


def apply_updates(params, updates):
    return tree_util.tree_map(lambda p, u: p + u, params, updates)
