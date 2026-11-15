"""Joint Adam training of predictor and amortized matcher."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gwsgp.data import N_CLASSES, sample_batch
from gwsgp.gw import gw_loss
from gwsgp.matcher import init_matcher, match, omega_bistochastic
from gwsgp.predictor import init_predictor, predict_adjacency


def init_params(key: jax.Array, n_nodes: int = 6) -> dict:
    k_p, k_m = jax.random.split(key)
    return {
        "predictor": init_predictor(k_p, n_nodes=n_nodes, n_classes=N_CLASSES),
        "matcher": init_matcher(k_m, n_nodes=n_nodes, n_classes=N_CLASSES),
    }


def sgp_loss_one(
    params,
    x,
    adj_star,
    tau: float = 0.1,
    n_sinkhorn: int = 24,
    omega_weight: float = 1.0,
):
    adj_hat = predict_adjacency(params["predictor"], x)
    plan = match(
        params["matcher"],
        x,
        adj_hat,
        adj_star,
        tau=tau,
        n_iters=n_sinkhorn,
    )
    matching = gw_loss(plan, adj_hat, adj_star)
    omega = omega_bistochastic(plan)
    return matching + omega_weight * omega, (matching, omega, adj_hat, plan)


def batch_loss(params, x, adj_star, tau=0.1, n_sinkhorn=24, omega_weight=1.0):
    losses, aux = jax.vmap(
        lambda xi, ai: sgp_loss_one(
            params, xi, ai, tau=tau, n_sinkhorn=n_sinkhorn, omega_weight=omega_weight
        )
    )(x, adj_star)
    matching = aux[0]
    return jnp.mean(losses), jnp.mean(matching)


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": 0}


def _adam_update(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda mi, g: b1 * mi + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda vi, g: b2 * vi + (1.0 - b2) * g * g, state["v"], grads)
    m_hat = jax.tree.map(lambda mi: mi / (1.0 - b1**t), m)
    v_hat = jax.tree.map(lambda vi: vi / (1.0 - b2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return params, {"m": m, "v": v, "t": t}


def train(
    key: jax.Array,
    n_steps: int = 200,
    batch_size: int = 16,
    n_nodes: int = 6,
    lr: float = 1e-2,
    tau: float = 0.1,
    n_sinkhorn: int = 24,
    omega_weight: float = 1.0,
    eval_size: int = 32,
):
    """Train predictor + matcher; return params and eval matching losses."""
    k_init, k_eval, k_train = jax.random.split(key, 3)
    params = init_params(k_init, n_nodes=n_nodes)
    opt = _adam_init(params)
    x_eval, adj_eval, _ = sample_batch(k_eval, eval_size, n_nodes=n_nodes)

    @jax.jit
    def eval_fn(params):
        _, matching = batch_loss(
            params, x_eval, adj_eval, tau=tau, n_sinkhorn=n_sinkhorn, omega_weight=omega_weight
        )
        return matching

    @jax.jit
    def step(params, opt, batch_key):
        x, adj_star, _ = sample_batch(batch_key, batch_size, n_nodes=n_nodes)

        def loss_fn(p):
            total, matching = batch_loss(
                p, x, adj_star, tau=tau, n_sinkhorn=n_sinkhorn, omega_weight=omega_weight
            )
            return total, matching

        (total, matching), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        params, opt = _adam_update(params, grads, opt, lr=lr)
        return params, opt, total, matching

    loss_before = float(eval_fn(params))
    keys = jax.random.split(k_train, n_steps)
    for i in range(n_steps):
        params, opt, _total, _matching = step(params, opt, keys[i])
    loss_after = float(eval_fn(params))
    return params, loss_before, loss_after
