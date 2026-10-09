"""Fixed synthetic batch. Painter first, then Thinker. Prints first and last loss."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp

from path_thinker.painter import init_painter
from path_thinker.thinker import init_thinker
from path_thinker.train import (
    alpha_bars,
    q_sample,
    thinker_loss,
    train_painter_step,
    train_thinker_step,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps-painter", type=int, default=20)
    p.add_argument("--steps-thinker", type=int, default=30)
    args = p.parse_args()

    d = 32
    height = width = 8
    n_spatial = height * width
    y_len = n_spatial + 1
    z_len = 4
    b = 4
    n_sup, H, L = 2, 2, 2

    key = jax.random.PRNGKey(0)
    k_p, k_t, k_x, k_e, k_n, k_w = jax.random.split(key, 6)
    painter = init_painter(k_p, c_in=1)
    thinker = init_thinker(k_t, d, y_len, z_len, c_skip=8)

    x0 = jax.random.normal(k_x, (b, height, width, 1))
    eps = jax.random.normal(k_e, x0.shape)
    ab = alpha_bars(100)
    t = jnp.full((b,), 10)
    x_t = q_sample(x0, eps, ab[t][:, None, None, None])

    noisy = q_sample(x0, jax.random.normal(k_n, x0.shape), ab[20])
    W_tok = jax.random.normal(k_w, (1, d)) * 0.02
    spatial = noisy.reshape(b, n_spatial, 1) @ W_tok
    onehot = jnp.zeros((b, 1, d)).at[:, 0, 0].set(1.0)
    c = jnp.concatenate([spatial, onehot], axis=1)
    batch = {"x0": x0, "x_t": x_t, "t": t, "c": c}

    for _ in range(args.steps_painter):
        painter, thinker = train_painter_step(painter, thinker, batch, lr=1e-2)

    kw = dict(
        n_sup=n_sup, H=H, L=L, n_spatial=n_spatial, height=height, width=width
    )
    first = float(thinker_loss(painter, thinker, batch, **kw))
    print("thinker loss first", first)
    for i in range(args.steps_thinker):
        painter, thinker = train_thinker_step(painter, thinker, batch, 1e-2, **kw)
        if i == args.steps_thinker - 1:
            last = float(thinker_loss(painter, thinker, batch, **kw))
    print("thinker loss last", last)

    out = Path(__file__).resolve().parents[1] / "smoke-results"
    out.mkdir(exist_ok=True)
    payload = {"first": first, "last": last}
    (out / "loss.json").write_text(json.dumps(payload) + "\n")
    print("wrote", out / "loss.json")


if __name__ == "__main__":
    main()
