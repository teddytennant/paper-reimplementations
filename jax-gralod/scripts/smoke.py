"""Tiny synthetic rain-ish image: init dummy restorer, 20 Adam steps, print losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_gralod.backbone import forward_train, init_restorer
from jax_gralod.model import GraLoDConfig


def synthetic_pair(key, size=16):
    k1, k2 = jax.random.split(key)
    yy, xx = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, size),
        jnp.linspace(0.0, 1.0, size),
        indexing="ij",
    )
    clean = 0.5 + 0.25 * jnp.sin(4.0 * jnp.pi * xx) * jnp.cos(4.0 * jnp.pi * yy)
    clean = jnp.stack([clean, 0.4 + 0.2 * yy, 0.6 - 0.2 * xx], axis=-1)
    clean = clean[None, ...]
    # diagonal rain-ish streaks
    streaks = 0.35 * jax.nn.relu(jnp.sin(18.0 * (xx + yy)))
    rain = jnp.clip(clean + streaks[..., None], 0.0, 1.0)
    noise = 0.02 * jax.random.normal(k2, rain.shape)
    degraded = jnp.clip(rain + noise, 0.0, 1.0)
    return degraded.astype(jnp.float32), clean.astype(jnp.float32)


def main():
    cfg = GraLoDConfig(
        encoder_channels=(8, 16, 32),
        decoder_channels=(8, 16, 32),
        lod_channels=8,
        warmup_steps=0,
    )
    key = jax.random.PRNGKey(0)
    k_init, k_data = jax.random.split(key)
    params = init_restorer(k_init, cfg)
    y, x = synthetic_pair(k_data, size=16)

    def loss_fn(p):
        loss, _aux = forward_train(p, y, x, cfg, step=0)
        return loss

    loss0 = float(loss_fn(params))
    print(f"train_loss_0={loss0:.6f}")

    lr = 3e-3
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_jit = jax.jit(loss_fn)

    p = params
    for t in range(1, 21):
        g = grad_fn(p)
        m = jax.tree.map(lambda mi, gi: b1 * mi + (1.0 - b1) * gi, m, g)
        v = jax.tree.map(lambda vi, gi: b2 * vi + (1.0 - b2) * gi * gi, v, g)
        mhat = jax.tree.map(lambda mi: mi / (1.0 - b1**t), m)
        vhat = jax.tree.map(lambda vi: vi / (1.0 - b2**t), v)
        p = jax.tree.map(lambda pi, mi, vi: pi - lr * mi / (jnp.sqrt(vi) + eps), p, mhat, vhat)

    loss20 = float(loss_jit(p))
    print(f"train_loss_20={loss20:.6f}")
    if not (jnp.isfinite(loss0) and jnp.isfinite(loss20)):
        raise SystemExit("non-finite loss")


if __name__ == "__main__":
    main()
