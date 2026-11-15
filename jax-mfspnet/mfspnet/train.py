"""Short training runs that produce the loss trajectories the estimators eat.

Alg. 3 line 10 trains a candidate for c validation-cutoff epochs and scores it
with VLE-EMA (Eq. 8). Table 3 fixes the search-time optimizer as Adam at a flat
1e-3, so that is the default here. The tests run at a much smaller scale than
the paper's 15 epochs at batch 256.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import Dataset
from .model import StackConfig, apply_network, init_network

LEARNING_RATE = 1e-3


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Per-epoch mean training and validation loss, plus final val accuracy."""

    train_losses: np.ndarray
    val_losses: np.ndarray
    val_accuracy: float


def cross_entropy(logits, labels):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels[:, None], 1))


def train_candidate(
    key,
    growth_rates,
    data: Dataset,
    cfg: StackConfig,
    epochs: int,
    batch_size: int = 64,
    learning_rate: float = LEARNING_RATE,
) -> Trajectory:
    """Train one decoded architecture and record its loss trajectory.

    Validation loss is measured once per epoch over the whole validation split,
    which is the inner mean of Eq. 8 with B_v equal to one full-set batch.
    """
    key, init_key = jax.random.split(key)
    params = init_network(init_key, growth_rates, data.in_channels, cfg)
    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    def loss_fn(p, x, y):
        return cross_entropy(apply_network(p, x, cfg), y)

    @jax.jit
    def update(p, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
        updates, state = opt.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    @jax.jit
    def evaluate(p, x, y):
        logits = apply_network(p, x, cfg)
        return cross_entropy(logits, y), jnp.mean(jnp.argmax(logits, -1) == y)

    n = data.x_train.shape[0]
    train_losses, val_losses = [], []
    val_acc = 0.0
    for _ in range(epochs):
        key, perm_key = jax.random.split(key)
        order = jax.random.permutation(perm_key, n)
        batch_losses = []
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            params, opt_state, loss = update(
                params, opt_state, data.x_train[idx], data.y_train[idx]
            )
            batch_losses.append(float(loss))
        vloss, vacc = evaluate(params, data.x_val, data.y_val)
        train_losses.append(float(np.mean(batch_losses)))
        val_losses.append(float(vloss))
        val_acc = float(vacc)
    return Trajectory(np.asarray(train_losses), np.asarray(val_losses), val_acc)
