"""LM loss and a tiny AdamW step. Not the paper's 50k-step TinyStories run."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax_prm.config import PRMConfig
from jax_prm.transformer import forward

Array = jax.Array
Params = dict[str, Any]


def _nll(logits: Array, targets: Array) -> Array:
    logp = jax.nn.log_softmax(logits, axis=-1)
    gather = jnp.take_along_axis(logp, targets[..., None], axis=-1)
    return -jnp.mean(gather)


def loss_fn(
    params: Params,
    tokens: Array,
    cfg: PRMConfig,
    *,
    key: Array | None = None,
    training: bool = False,
) -> Array:
    """Next-token LM loss. If aux_lambda > 0, add MSE(stop_grad(pool H_lower), W_aux pool M).

    The paper's auxiliary objective is underspecified in the PDF; this MSE is a
    stand-in so λ can be non-zero. Default λ=0 uses only L_LM.
    """
    logits, info = forward(params, tokens, cfg, key=key, training=training)
    lm = _nll(logits[:, :-1], tokens[:, 1:])
    if cfg.aux_lambda > 0:
        pooled_m = jnp.mean(info["memory"], axis=1)
        pred = pooled_m @ params["prm"]["W_aux"]
        target = jax.lax.stop_gradient(jnp.mean(info["h_lower"], axis=1))
        aux = jnp.mean((pred - target) ** 2)
        return lm + cfg.aux_lambda * aux
    return lm


def adamw_init(params: Params) -> Params:
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "step": jnp.array(0, dtype=jnp.int32),
    }


def adamw_step(
    params: Params,
    opt_state: Params,
    tokens: Array,
    cfg: PRMConfig,
    *,
    key: Array | None = None,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Params, Params, Array]:
    """One AdamW update on a token batch. Decays rank-2+ tensors only."""

    def _loss(p):
        return loss_fn(p, tokens, cfg, key=key, training=True)

    loss, grads = jax.value_and_grad(_loss)(params)
    step = opt_state["step"] + 1
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), opt_state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda mi: mi / (1.0 - b1 ** step), m)
    vhat = jax.tree_util.tree_map(lambda vi: vi / (1.0 - b2 ** step), v)

    def _update(p, mh, vh):
        adam = mh / (jnp.sqrt(vh) + eps)
        if p.ndim >= 2:
            adam = adam + weight_decay * p
        return p - lr * adam

    new_params = jax.tree_util.tree_map(_update, params, mhat, vhat)
    new_state = {"m": m, "v": v, "step": step}
    return new_params, new_state, loss
