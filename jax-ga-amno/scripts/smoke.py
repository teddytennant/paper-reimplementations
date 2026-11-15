"""CPU/GPU smoke: 16x16 field, one forward, a few Adam steps on toy Poisson."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_ga_amno.model import GAAMNOConfig, ga_amno_forward, init_ga_amno


def _adam(params, grads, state, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda mm, g: b1 * mm + (1.0 - b1) * g, m, grads)
    v = jax.tree.map(lambda vv, g: b2 * vv + (1.0 - b2) * g * g, v, grads)
    mhat = jax.tree.map(lambda mm: mm / (1.0 - b1**t), m)
    vhat = jax.tree.map(lambda vv: vv / (1.0 - b2**t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return params, (m, v, t)


def main() -> None:
    height = width = 16
    cfg = GAAMNOConfig(height=height, width=width, hidden=8, rank=2, n_layers=2)
    key = jax.random.PRNGKey(0)
    params = init_ga_amno(key, cfg)
    field = jax.random.normal(jax.random.PRNGKey(1), (height, width))
    out = ga_amno_forward(params, field, cfg)
    mean_dx = float(jnp.mean(jnp.abs(out["dx"])))
    mean_m = float(jnp.mean(jnp.linalg.norm(out["m"], axis=-1)))
    print(f"n={height}x{width} layers={cfg.n_layers} hidden={cfg.hidden}")
    print(f"mean_|dx|={mean_dx:.6g}")
    print(f"mean_||m||={mean_m:.6g}")

    ys = jnp.linspace(0.0, 1.0, height)
    xs = jnp.linspace(0.0, 1.0, width)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    u = jnp.sin(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * yy)
    f = -8.0 * (jnp.pi**2) * u

    def loss_fn(p):
        pred = ga_amno_forward(p, f, cfg)["out"][..., 0]
        return jnp.mean((pred - u) ** 2)

    loss0 = float(loss_fn(params))
    m0 = jax.tree.map(jnp.zeros_like, params)
    v0 = jax.tree.map(jnp.zeros_like, params)
    state = (m0, v0, 0)
    step = jax.jit(jax.value_and_grad(loss_fn))
    p = params
    loss = loss0
    for _ in range(20):
        loss, grads = step(p)
        p, state = _adam(p, grads, state)
    loss20 = float(loss_fn(p))
    print(f"train_loss_0={loss0:.6g} train_loss_20={loss20:.6g}")
    if not jnp.isfinite(loss20):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
