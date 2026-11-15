"""Toy AXON classification on a synthetic C x T grid."""

import jax
import jax.numpy as jnp

from aaa.model import axon_classify, init_axon_params


C, T, D, HEADS, CLASSES = 4, 8, 32, 4, 2
BATCH = 32
STEPS = 200
LR = 3e-3
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def make_batch(key, n):
    """Class depends on a channel-time pattern: sign of x[c=0, t=3] vs x[c=2, t=6]."""
    x = jax.random.normal(key, (n, C, T, D))
    score = x[:, 0, 3, 0] - x[:, 2, 6, 0]
    y = (score > 0).astype(jnp.int32)
    cue = jnp.where(y == 1, 1.5, -1.5)
    x = x.at[:, 0, 3, 0].add(cue)
    x = x.at[:, 2, 6, 0].add(-cue)
    return x, y


def loss_fn(params, x, y):
    logits = axon_classify(params, x, num_heads=HEADS)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=-1))


def accuracy(params, x, y):
    logits = axon_classify(params, x, num_heads=HEADS)
    pred = jnp.argmax(logits, axis=-1)
    return jnp.mean((pred == y).astype(jnp.float32))


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(params, grads, state):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(
        lambda m, g: BETA1 * m + (1.0 - BETA1) * g, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v, g: BETA2 * v + (1.0 - BETA2) * jnp.square(g), state["v"], grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - BETA1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - BETA2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - LR * mh / (jnp.sqrt(vh) + EPS), params, mhat, vhat
    )
    return params, {"m": m, "v": v, "t": t}


def main():
    key = jax.random.PRNGKey(0)
    k_init, k_data, k_train = jax.random.split(key, 3)
    params = init_axon_params(
        k_init, c=C, t=T, d=D, num_blocks=2, num_classes=CLASSES, gate="gated"
    )
    x, y = make_batch(k_data, BATCH)
    opt_state = adam_init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params, opt_state = adam_update(params, grads, opt_state)
        return params, opt_state, loss

    loss0 = float(loss_fn(params, x, y))
    print(f"loss_before={loss0:.4f}")
    for _ in range(STEPS):
        k_train, k_step = jax.random.split(k_train)
        xb, yb = make_batch(k_step, BATCH)
        params, opt_state, loss = step(params, opt_state, xb, yb)
    loss1 = float(loss)
    acc = float(accuracy(params, x, y))
    print(f"loss_after={loss1:.4f}")
    print(f"train_acc={acc:.3f}")
    if not (loss1 < loss0):
        raise SystemExit("loss did not drop")


if __name__ == "__main__":
    main()
