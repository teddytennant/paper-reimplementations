"""Unroll a few trials and print mean reward / MSE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_astrocyte_ha.dynamics import init_arnn, init_nrnn
from jax_astrocyte_ha.env import Config, make_nrnn_batch, tile_contexts
from jax_astrocyte_ha.train import a2c_train_step, collect_a2c_rollout, hybrid_rollout, nrnn_train_step


def main():
    cfg = Config(M=2, Nx=16, Nh=8, h_std=0.2)
    key = jax.random.PRNGKey(0)
    k_n, k_a, k_b, k_r = jax.random.split(key, 4)
    nrnn = init_nrnn(k_n, Nx=cfg.Nx, M=cfg.M, h_std=cfg.h_std)
    arnn = init_arnn(k_a, Nh=cfg.Nh, M=cfg.M, No=cfg.No, hidden=cfg.hidden)
    batch = make_nrnn_batch(k_b, cfg, batch_size=8)
    step_n = jax.jit(lambda p, b: nrnn_train_step(p, b, cfg.tau_x, 0.05))
    for _ in range(8):
        nrnn, nloss, _ = step_n(nrnn, batch)

    contexts = tile_contexts([0, 1], [5, 5])
    thetas = np.linspace(0.3, 4.0, len(contexts))
    h0, actions, rewards, _ = collect_a2c_rollout(
        nrnn, arnn, cfg, k_r, contexts, thetas, sample_actions=True
    )
    arnn, aloss, _ = a2c_train_step(arnn, h0, actions, rewards, cfg.tau_h, cfg, lr=0.01)

    out = hybrid_rollout(nrnn, arnn, cfg, contexts, thetas)
    mean_reward = float(jnp.mean(out["rewards"]))
    mean_mse = float(jnp.mean(out["mses"]))
    print(f"mean_reward={mean_reward:.6f}")
    print(f"mean_mse={mean_mse:.6f}")
    print(f"nrnn_mse={float(nloss):.6f}")
    print(f"a2c_loss={float(aloss):.6f}")
    if not (np.isfinite(mean_reward) and np.isfinite(mean_mse)):
        raise SystemExit("non-finite rollout")


if __name__ == "__main__":
    main()
