"""Two-level Painter. Skips are additions. Control is a zero-init residual."""

import jax
import jax.numpy as jnp

C1 = 8
C2 = 16
T_DIM = 16


def init_painter(key, c_in, scale=0.02):
    keys = jax.random.split(key, 6)

    def n(k, shape):
        return jax.random.normal(k, shape) * scale

    return {
        "W1": n(keys[0], (c_in, C1)),
        "b1": jnp.zeros((C1,)),
        "W2": n(keys[1], (C1, C2)),
        "b2": jnp.zeros((C2,)),
        "W_up": n(keys[2], (C2, C1)),
        "b_up": jnp.zeros((C1,)),
        "Wout": n(keys[3], (C1, c_in)),
        "bout": jnp.zeros((c_in,)),
        "W_t1": n(keys[4], (T_DIM, C1)),
        "b_t1": jnp.zeros((C1,)),
        "W_t2": n(keys[5], (T_DIM, C2)),
        "b_t2": jnp.zeros((C2,)),
    }


def sinusoidal(t, dim):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    ang = t.astype(jnp.float32)[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def pool2(x):
    b, h, w, c = x.shape
    x = x.reshape(b, h // 2, 2, w // 2, 2, c)
    return x.mean(axis=(2, 4))


def repeat2(x):
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


def painter_forward(params, x, t, control=None, adapters=None, return_skips=False):
    """x0 prediction. control is (B, H, W, C1). adapters are the zero-init maps.

    A missing control or missing adapters adds nothing. level1 skip is
    level1 + control @ W_a1 + b_a1. level2 gets the pooled control through
    W_a2 before the upsample.
    """
    t = t.astype(jnp.float32)
    temb = sinusoidal(t, T_DIM)
    te1 = temb @ params["W_t1"] + params["b_t1"]
    te2 = temb @ params["W_t2"] + params["b_t2"]

    level1 = x @ params["W1"] + params["b1"]
    level1 = level1 + te1[:, None, None, :]
    level2 = pool2(level1) @ params["W2"] + params["b2"]
    level2 = level2 + te2[:, None, None, :]

    if control is None or adapters is None:
        control1 = jnp.zeros_like(level1)
        control2 = jnp.zeros_like(level2)
    else:
        control1 = control @ adapters["W_a1"] + adapters["b_a1"]
        control2 = pool2(control) @ adapters["W_a2"] + adapters["b_a2"]

    level2 = level2 + control2
    up = repeat2(level2) @ params["W_up"] + params["b_up"]
    skip = level1 + control1
    pred = (skip + up) @ params["Wout"] + params["bout"]
    if return_skips:
        return pred, skip, level2
    return pred
