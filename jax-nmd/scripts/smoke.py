"""Tiny synthetic 2-port train loop. Prints train_loss_0 and train_loss_N."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_nmd.config import NMDConfig
from jax_nmd.train import init_adamw, init_params, make_synthetic_batch, predict_loss, train_step

N_STEPS = 40


def main():
    cfg = NMDConfig(k_poles=4, d_t=32, n_tokens=4, hidden=32, conv_mid=8)
    key = jax.random.PRNGKey(0)
    k_init, k_data = jax.random.split(key)
    params = init_params(k_init, cfg)
    batch = make_synthetic_batch(k_data, cfg, batch_size=1, n_ports=2, n_freq=12)
    loss0, _ = predict_loss(params, batch, cfg)
    loss0 = float(loss0)
    print(f"train_loss_0={loss0:.6f}")

    opt = init_adamw(params)
    p, o = params, opt
    for _ in range(N_STEPS):
        p, o, _ = train_step(p, o, batch, cfg)
    loss_n, _ = predict_loss(p, batch, cfg)
    loss_n = float(loss_n)
    print(f"train_loss_N={loss_n:.6f}")

    if not (jnp.isfinite(loss0) and jnp.isfinite(loss_n)):
        raise SystemExit("non-finite loss")
    if not (loss_n < loss0):
        raise SystemExit("loss did not decrease")


if __name__ == "__main__":
    main()
