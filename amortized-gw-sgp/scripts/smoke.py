"""Short toy SGP training run."""

from __future__ import annotations

import jax

from gwsgp.train import train


def main():
    key = jax.random.PRNGKey(0)
    _params, loss_before, loss_after = train(
        key,
        n_steps=200,
        batch_size=16,
        n_nodes=6,
        lr=1e-2,
        tau=0.1,
        n_sinkhorn=20,
        omega_weight=1.0,
        eval_size=24,
    )
    print(f"loss_before: {loss_before}")
    print(f"loss_after: {loss_after}")


if __name__ == "__main__":
    main()
