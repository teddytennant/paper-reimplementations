"""NLL training, greedy generation, Adam (paper: Adam, lr=1e-2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_vqt.model import VQTConfig, batch_nll, next_token_probs


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(params, grads, state, lr: float = 1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1.0 - b2) * jnp.square(g), state["v"], grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t}


def make_step(config: VQTConfig, lr: float = 1e-2):
    def step(params, opt_state, contexts, targets):
        def loss_fn(p):
            return batch_nll(p, contexts, targets, config)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt_state = adam_update(params, grads, opt_state, lr=lr)
        return params, opt_state, loss

    return jax.jit(step)


def greedy_generate(params, context, n_tokens: int, config: VQTConfig):
    """Greedy next-token generation from a length-c context (paper §4)."""
    ctx = jnp.asarray(context, dtype=jnp.int32)
    assert ctx.shape == (config.context,)
    out = []
    for _ in range(n_tokens):
        probs = next_token_probs(params, ctx, config)
        tok = jnp.argmax(probs).astype(jnp.int32)
        out.append(tok)
        ctx = jnp.concatenate([ctx[1:], tok[None]])
    return jnp.stack(out)
