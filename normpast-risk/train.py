"""Seeded synthetic training loop for NormPaST-Risk.

Uses a tiny configuration (T=64, gamma=16, d=16, one SSM layer, two experts,
three tasks) and 30 Adam steps. Seed 0. Dropout stays off.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import optax

from normpast_risk.loss import total_loss
from normpast_risk.model import apply, init_params


def make_synthetic_batch(key, batch_size: int = 8, length: int = 64, n_tasks: int = 3):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    kx, kq, kt = jax.random.split(key, 3)
    x = jax.random.normal(kx, (batch_size, length, 4), dtype=jnp.float32)
    q = jax.random.bernoulli(kq, 0.5, (batch_size, length)).astype(jnp.float32)
    task_id = jax.random.randint(kt, (batch_size,), 0, n_tasks)
    y = jnp.tile(jnp.asarray([0.0, 1.0], dtype=jnp.float32), batch_size // 2)
    return {"x": x, "q": q, "task_id": task_id, "y": y}


def run_training(steps: int = 30, seed: int = 0, lr: float = 1e-3):
    """Run Adam on one synthetic batch. Returns pre-update losses and the final loss."""
    key = jax.random.PRNGKey(seed)
    key_params, key_batch = jax.random.split(key)
    batch = make_synthetic_batch(key_batch)
    params = init_params(
        key_params,
        d=16,
        n_layers=1,
        n_experts=2,
        n_tasks=3,
        in_dim=4,
        kernel_sizes=(3, 7),
        dilations=(1, 2),
    )
    gamma = 16
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, batch):
        outputs = apply(
            p,
            batch["x"],
            batch["q"],
            batch["task_id"],
            gamma=gamma,
            dilations=(1, 2),
            dropout_rate=0.0,
        )
        return total_loss(outputs, batch["y"])["loss"]

    @jax.jit
    def step(p, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(p, batch)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    losses = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, batch)
        losses.append(loss)
    final = loss_fn(params, batch)
    return {
        "losses": jnp.stack(losses),
        "loss_after": final,
        "params": params,
        "batch": batch,
    }


def main():
    result = run_training()
    losses = result["losses"]
    print(f"step 0 loss {float(losses[0]):.6f}")
    print(f"step 29 loss {float(losses[29]):.6f}")


if __name__ == "__main__":
    main()
