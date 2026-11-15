"""Tiny ε-prediction MLP for the 2D learned-score demo."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from intrinsic_atlas.gaussian import tweedie_x0_from_eps
from intrinsic_atlas.schedules import VPSchedule

TIME_DIM = 32


def _linear(key, in_dim, out_dim, scale=1.0):
    w_key, _ = jax.random.split(key)
    w = scale * jax.random.normal(w_key, (in_dim, out_dim)) / jnp.sqrt(in_dim)
    b = jnp.zeros((out_dim,))
    return w, b


def init_model(key, in_dim=2, hidden=64, time_dim=TIME_DIM, n_hidden=2):
    keys = jax.random.split(key, n_hidden + 1)
    layers = [_linear(keys[0], in_dim + time_dim, hidden)]
    for i in range(1, n_hidden):
        layers.append(_linear(keys[i], hidden, hidden))
    layers.append(_linear(keys[-1], hidden, in_dim, scale=0.01))
    return tuple(layers)


def time_embedding(t, dim=TIME_DIM):
    """Sinusoidal embedding of integer timestep t (scalar or leading-batch array)."""
    t = jnp.asarray(t)
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    ang = t.astype(freqs.dtype)[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def predict_eps(params, x, t, time_dim=TIME_DIM):
    emb = time_embedding(t, time_dim)
    if emb.ndim == 1 and x.ndim >= 2:
        emb = jnp.broadcast_to(emb, x.shape[:-1] + (emb.shape[-1],))
    h = jnp.concatenate([x, emb], axis=-1)
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def noise_pred_loss(params, x0, t, eps, schedule: VPSchedule):
    alpha_bar = schedule.alphas_cumprod[t]
    while alpha_bar.ndim < x0.ndim:
        alpha_bar = alpha_bar[..., None]
    x_t = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * eps
    if x0.ndim == 2 and t.ndim == 1:
        pred = jax.vmap(lambda xi, ti: predict_eps(params, xi, ti))(x_t, t)
    else:
        pred = predict_eps(params, x_t, t)
    return jnp.mean((pred - eps) ** 2)


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros, jnp.array(0, dtype=jnp.int32)


def adam_update(params, grads, opt, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v, step = opt
    step = step + 1
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), v, grads)
    bc1 = 1.0 - b1 ** step.astype(jnp.float32)
    bc2 = 1.0 - b2 ** step.astype(jnp.float32)
    params = jax.tree_util.tree_map(
        lambda p, mi, vi: p - lr * (mi / bc1) / (jnp.sqrt(vi / bc2) + eps),
        params,
        m,
        v,
    )
    return params, (m, v, step)


def train_step(params, opt, batch_x0, key, schedule, lr):
    """One Adam step on a batch of clean samples."""
    key_t, key_eps, key = jax.random.split(key, 3)
    T = schedule.betas.shape[0]
    bsz = batch_x0.shape[0]
    t = jax.random.randint(key_t, (bsz,), 0, T)
    eps = jax.random.normal(key_eps, batch_x0.shape, dtype=batch_x0.dtype)

    def loss_fn(p):
        return noise_pred_loss(p, batch_x0, t, eps, schedule)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, opt = adam_update(params, grads, opt, lr)
    return params, opt, loss, key


def make_learned_x0_fn(params, schedule: VPSchedule):
    """predict_x0(x_t, t) from a trained ε-network (t is a scalar index)."""

    def predict_x0(x_t, t):
        alpha_bar = schedule.alphas_cumprod[t]
        eps = predict_eps(params, x_t, t)
        return tweedie_x0_from_eps(x_t, alpha_bar, eps)

    return predict_x0
