"""Adam training of GraphSAGE encoder + variational last layer."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from gvbll.elbo import map_nll, vbll_neg_elbo
from gvbll.encoder import encode, init_graphsage, to_dense_adj
from gvbll.vbll import init_vbll, predict_map, std_from_log_std


def init_gvbll(key, in_dim, n_classes, hidden_dim=16, embed_dim=16, prior_std=1.0):
    k_enc, _k_vbll = jax.random.split(key)
    return {
        "encoder": init_graphsage(k_enc, in_dim, hidden_dim, embed_dim),
        "vbll": init_vbll(n_classes, embed_dim, prior_std=prior_std),
    }


def adam_init(params):
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "t": jnp.zeros((), dtype=jnp.int32),
    }


def adam_update(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda mi: mi / (1.0 - b1**t), m)
    vhat = jax.tree.map(lambda vi: vi / (1.0 - b2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


def kl_beta(step, anneal_steps):
    """Linear anneal of beta from 0 to 1 over the first `anneal_steps` steps."""
    anneal_steps = jnp.asarray(anneal_steps, dtype=jnp.float32)
    step = jnp.asarray(step, dtype=jnp.float32)
    return jnp.where(
        anneal_steps > 0,
        jnp.clip((step + 1.0) / anneal_steps, 0.0, 1.0),
        jnp.ones_like(step),
    )


@partial(jax.jit, static_argnames=("n_samples",))
def train_step(
    params,
    opt_state,
    x,
    adj,
    y,
    key,
    beta,
    n_samples=8,
    prior_std=1.0,
    lr=1e-2,
    temperature=1.0,
):
    def loss_fn(params):
        h = encode(params["encoder"], x, adj)
        return vbll_neg_elbo(
            params["vbll"],
            h,
            y,
            key,
            n_samples=n_samples,
            beta=beta,
            prior_std=prior_std,
            temperature=temperature,
        )

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt_state = adam_update(params, grads, opt_state, lr=lr)
    return params, opt_state, loss


def train(
    params,
    x,
    adj,
    y,
    key,
    steps=120,
    lr=1e-2,
    n_samples=8,
    prior_std=1.0,
    anneal_steps=None,
    temperature=1.0,
):
    """Run Adam for `steps` iterations. Returns (params, losses)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y).astype(jnp.int32)
    adj = to_dense_adj(adj, x.shape[0])
    if anneal_steps is None:
        anneal_steps = max(steps // 3, 1)
    opt_state = adam_init(params)

    def body(carry, t):
        params, opt_state, key = carry
        key, k = jax.random.split(key)
        beta = kl_beta(t, anneal_steps)
        params, opt_state, loss = train_step(
            params,
            opt_state,
            x,
            adj,
            y,
            k,
            beta,
            n_samples=n_samples,
            prior_std=prior_std,
            lr=lr,
            temperature=temperature,
        )
        return (params, opt_state, key), loss

    (params, _opt_state, _key), losses = jax.lax.scan(
        body, (params, opt_state, key), jnp.arange(steps)
    )
    return params, losses


def evaluate(params, x, adj, y, temperature=1.0):
    h = encode(params["encoder"], x, adj)
    nll = map_nll(params["vbll"]["mean"], h, y, temperature=temperature)
    probs = predict_map(params["vbll"]["mean"], h, temperature=temperature)
    acc = jnp.mean((jnp.argmax(probs, axis=-1) == y.astype(jnp.int32)).astype(jnp.float32))
    return {"nll": nll, "acc": acc, "probs": probs, "h": h}


def embeddings(params, x, adj):
    return encode(params["encoder"], x, adj)


def last_layer_std(params):
    return std_from_log_std(params["vbll"]["log_std"])
