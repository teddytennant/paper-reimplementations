"""Tiny synthetic overfitting loop for LPS + 2 LAM blocks + linear head."""

import json
import os
import sys
from pathlib import Path

if os.environ.get("POINTLAM_CPU", "") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.is_dir():
    sys.path.insert(0, str(src))

import jax
import jax.numpy as jnp

from pointlam.lam import apply_pointlam, init_pointlam


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return zeros, zeros, jnp.array(0, dtype=jnp.int32)


def _adam_update(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda mi, g: b1 * mi + (1.0 - b1) * g, m, grads)
    v = jax.tree.map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), v, grads)
    mhat = jax.tree.map(lambda mi: mi / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda vi: vi / (1.0 - b2 ** t), v)
    params = jax.tree.map(
        lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), params, mhat, vhat
    )
    return params, (m, v, t)


def _make_cloud(key, n, n_centers=4):
    k1, k2, k3 = jax.random.split(key, 3)
    centers = jax.random.uniform(k1, (n_centers, 3), minval=0.15, maxval=0.85)
    assign = jax.random.randint(k2, (n,), 0, n_centers)
    noise = 0.04 * jax.random.normal(k3, (n, 3))
    points = jnp.clip(centers[assign] + noise, 0.0, 1.0)
    d = jnp.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    target = d.min(axis=-1)
    return points, target


def main():
    cpu = os.environ.get("POINTLAM_CPU", "") == "1"
    if cpu:
        n, m, dim, steps, cell_size = 64, 24, 8, 40, 0.25
    else:
        n, m, dim, steps, cell_size = 256, 64, 16, 80, 0.2

    key = jax.random.PRNGKey(0)
    k_data, k_init, k_lps = jax.random.split(key, 3)
    points, target = _make_cloud(k_data, n)
    params = init_pointlam(k_init, dim=dim, n_blocks=2, state=8)

    def loss_fn(p):
        _pts, _feats, idx, logits = apply_pointlam(
            p, points, k_lps, m=m, cell_size=cell_size
        )
        pred = logits[:, 0]
        y = target[idx]
        return jnp.mean((pred - y) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    opt = _adam_init(params)
    loss0 = None
    loss_t = None
    for _step in range(steps):
        loss, grads = loss_and_grad(params)
        params, opt = _adam_update(params, grads, opt, lr=2e-2)
        if loss0 is None:
            loss0 = float(loss)
        loss_t = float(loss)

    ok = bool(loss_t < 0.8 * loss0)
    print(json.dumps({"loss0": loss0, "lossT": loss_t, "ok": ok}))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
