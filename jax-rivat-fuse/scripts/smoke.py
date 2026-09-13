"""Tiny synthetic train of RiVaT-Fuse fusion params + quadratic head."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from rivat_fuse.model import fuse_one, init_params, predict


D = 8
S_DIM = 3
N = 64
STEPS = 200
LR = 0.02
N_CLASSES = 1


def make_data(key):
    kI, kM, kw = jax.random.split(key, 3)
    z_I = jax.random.normal(kI, (N, D))
    z_M = jax.random.normal(kM, (N, D))
    # Label depends on a known linear mix plus a multiplicative interaction.
    w = jax.random.normal(kw, (D,))
    logit = (z_I + z_M) @ w + jnp.sum(z_I * z_M, axis=-1)
    y = jnp.sign(logit)
    y = jnp.where(y == 0, 1.0, y)
    s = 0.5 * jnp.ones((N, S_DIM), dtype=jnp.float32)
    return z_I, z_M, s, y


def batch_loss(params, z_I, z_M, s, y):
    def one(zi, zm, ss, yy):
        score = predict(params, zi, zm, s=ss)[0]
        # Softplus hinge on signed labels: log(1 + exp(-y f)).
        return jax.nn.softplus(-yy * score)

    return jnp.mean(jax.vmap(one)(z_I, z_M, s, y))


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_data, k_params = jax.random.split(key)
    z_I, z_M, s, y = make_data(k_data)
    params = init_params(
        k_params,
        d=D,
        s_dim=S_DIM,
        r_rank=4,
        K=2,
        r_k=4,
        head_rank=2,
        n_classes=N_CLASSES,
        lam=1e-2,
    )

    loss_fn = lambda p: batch_loss(p, z_I, z_M, s, y)
    loss_before = float(loss_fn(params))

    def _trainable(p):
        return {k: p[k] for k in ("trust", "interaction", "head")}

    @jax.jit
    def step(p):
        g = jax.grad(lambda q: loss_fn({**p, **q}))(_trainable(p))
        updated = jax.tree_util.tree_map(lambda x, dx: x - LR * dx, _trainable(p), g)
        return {**p, **updated}

    p = params
    for _ in range(STEPS):
        p = step(p)

    loss_after = float(loss_fn(p))
    print(f"loss_before={loss_before:.6f}")
    print(f"loss_after={loss_after:.6f}")

    h = fuse_one(p, z_I[0], z_M[0], s=s[0])
    if not jnp.all(jnp.isfinite(h)):
        print("fused state is not finite", file=sys.stderr)
        sys.exit(1)
    if not (loss_after < loss_before):
        print("loss did not drop", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
