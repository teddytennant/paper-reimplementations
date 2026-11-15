"""Recursive Thinker: latent z and answer y inside one denoising step.

One shared block. z is updated from concat(c, y, z) and is never passed to
the decoder. y is updated from concat(y, z). Incoming state is detached at
each supervision step, and again between inner cycles except the last one.
"""

import jax
import jax.numpy as jnp


def init_thinker(key, d, y_len, z_len, c_skip, scale=0.02):
    kq, kk, kv, ko, ky, kz, kd = jax.random.split(key, 7)
    return {
        "Wq": jax.random.normal(kq, (d, d)) * scale,
        "Wk": jax.random.normal(kk, (d, d)) * scale,
        "Wv": jax.random.normal(kv, (d, d)) * scale,
        "Wo": jax.random.normal(ko, (d, d)) * scale,
        "y0": jax.random.normal(ky, (y_len, d)) * scale,
        "z0": jax.random.normal(kz, (z_len, d)) * scale,
        "W_dec": jax.random.normal(kd, (d, c_skip)) * scale,
        "b_dec": jnp.zeros((c_skip,)),
        # ControlNet adapters. Zero so a fresh Thinker does not move the Painter.
        # They live here, not in the frozen Painter, so the Thinker can train them.
        "W_a1": jnp.zeros((c_skip, c_skip)),
        "b_a1": jnp.zeros((c_skip,)),
        "W_a2": jnp.zeros((c_skip, c_skip * 2)),
        "b_a2": jnp.zeros((c_skip * 2,)),
    }


def block(params, tokens):
    d = tokens.shape[-1]
    q = tokens @ params["Wq"]
    k = tokens @ params["Wk"]
    v = tokens @ params["Wv"]
    scale = jnp.sqrt(jnp.asarray(d, dtype=tokens.dtype))
    attn = jax.nn.softmax(q @ jnp.swapaxes(k, -1, -2) / scale, axis=-1)
    return tokens + (attn @ v) @ params["Wo"]


def latent_update(params, c, y, z):
    tokens = jnp.concatenate([c, y, z], axis=1)
    out = block(params, tokens)
    return out[:, -z.shape[1] :]


def answer_update(params, y, z):
    tokens = jnp.concatenate([y, z], axis=1)
    out = block(params, tokens)
    return out[:, : y.shape[1]]


def think_once(params, c, y, z, H, L, inject_after_h=None):
    """H cycles of L latent updates and one answer update.

    After every cycle except the last, y and z are detached. inject_after_h
    maps a cycle index to a tensor added to y after that cycle's answer update,
    before the detach. Tests use it to mark an early write.
    """
    for h in range(H):
        for _ in range(L):
            z = latent_update(params, c, y, z)
        y = answer_update(params, y, z)
        if inject_after_h is not None and h in inject_after_h:
            y = y + inject_after_h[h]
        if h < H - 1:
            y = jax.lax.stop_gradient(y)
            z = jax.lax.stop_gradient(z)
    return y, z


def recurse_from(params, c, y, z, n_sup, H, L, inject_after_h=None):
    """N_sup supervision steps. Incoming y and z are detached at each step."""
    ys = []
    zs = []
    for _ in range(n_sup):
        y = jax.lax.stop_gradient(y)
        z = jax.lax.stop_gradient(z)
        y, z = think_once(params, c, y, z, H, L, inject_after_h=inject_after_h)
        ys.append(y)
        zs.append(z)
    return ys, zs


def recurse(params, c, n_sup, H, L, inject_after_h=None):
    b = c.shape[0]
    y = jnp.broadcast_to(params["y0"], (b,) + params["y0"].shape)
    z = jnp.broadcast_to(params["z0"], (b,) + params["z0"].shape)
    return recurse_from(params, c, y, z, n_sup, H, L, inject_after_h=inject_after_h)


def spatial_tokens(y, n_spatial):
    return y[:, :n_spatial]


def project_tokens(params, tokens):
    return tokens @ params["W_dec"] + params["b_dec"]


def decode(params, y, n_spatial, height, width):
    """Map spatial tokens of y to a feature map. z is not an input."""
    spatial = spatial_tokens(y, n_spatial)
    cells = project_tokens(params, spatial)
    b = y.shape[0]
    return cells.reshape(b, height, width, -1)
