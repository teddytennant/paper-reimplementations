"""Training for the toy classifier. CPU only, a few seconds.

The whole loop is one `lax.scan`, so the steps compile once instead of
dispatching one at a time; on a model this small that is most of the runtime.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import optax

from .model import ModelConfig, batched_logits, init_params
from .task import make_batch


def loss_fn(cfg, params, tokens, labels):
    logits = batched_logits(cfg, params, tokens, None)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 5))
def _run(key, cfg, steps, batch_size, lr, dtype):
    key_init, key_data = jax.random.split(key)
    params = init_params(key_init, cfg, dtype)
    opt = optax.adam(lr)
    state = opt.init(params)

    def step(carry, i):
        params, state = carry
        tokens, labels = make_batch(jax.random.fold_in(key_data, i), batch_size,
                                    cfg.seq_len, cfg.vocab_size)
        loss, grads = jax.value_and_grad(lambda p: loss_fn(cfg, p, tokens, labels))(params)
        updates, state = opt.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), loss

    (params, _), losses = jax.lax.scan(step, (params, state), jnp.arange(steps))
    return params, losses


def train(
    key,
    cfg: ModelConfig | None = None,
    steps: int = 800,
    batch_size: int = 128,
    lr: float = 3e-3,
    dtype=jnp.float32,
):
    """Train on the synthetic task and return (params, held-out accuracy)."""
    cfg = cfg or ModelConfig()
    params, _losses = _run(key, cfg, steps, batch_size, lr, dtype)
    eval_tokens, eval_labels = make_batch(jax.random.fold_in(key, 10_000), 1024,
                                          cfg.seq_len, cfg.vocab_size)
    acc = (batched_logits(cfg, params, eval_tokens, None).argmax(-1) == eval_labels).mean()
    return params, float(acc)
