"""Tiny FCx train on synthetic data; print metrics; exit 0."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from jax_fcx.classifier import accuracy, train_classifier
from jax_fcx.train import FCxConfig, evaluate_cf, make_synthetic_data, train_fcx


def main() -> int:
    data = make_synthetic_data(n=256, d=8, seed=0)
    x = jnp.asarray(data["x"])
    y = jnp.asarray(data["y"])
    key = jax.random.PRNGKey(0)
    key, k_clf, k_fcx = jax.random.split(key, 3)
    clf_params, _ = train_classifier(k_clf, x, y, hidden=16, steps=80, batch_size=64)
    acc = float(accuracy(clf_params, x, y))
    cfg = FCxConfig(hidden=16, n_linear=3, steps=15, batch_size=32, lr=1e-2, z_dim=8)
    vae_params, logs = train_fcx(
        k_fcx,
        x,
        clf_params,
        config=cfg,
        adjacency=jnp.asarray(data["A"]),
        mutable_idx=jnp.asarray(data["mutable_idx"]),
    )
    metrics = evaluate_cf(
        vae_params,
        clf_params,
        logs["x"][:32],
        logs["y_target"][:32],
        logs["mutable_idx"],
        binary=False,
    )
    print("classifier_acc", round(acc, 4))
    print("loss0", round(logs["loss"][0], 4), "lossN", round(logs["loss"][-1], 4))
    print("metrics", {k: round(v, 4) for k, v in metrics.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
