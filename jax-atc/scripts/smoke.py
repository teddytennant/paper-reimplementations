"""Tiny n=4 ATC curriculum; prints per-phase sign accuracy."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from jax_atc.model import ATCConfig, init_params
from jax_atc.train import run_curriculum


def main():
    n, d = 4, 32
    config = ATCConfig.tiny(n=n, d=d)
    key = jax.random.PRNGKey(0)
    key, pkey = jax.random.split(key)
    params = init_params(pkey, config)
    _, history = run_curriculum(
        params,
        key,
        config,
        n_phases=2,
        lr=1e-2,
        batch_size=64,
        max_steps=400,
        loss_threshold=0.02,
        eval_every=20,
    )
    for info in history:
        print(
            f"phase {info['phase']}: acc={info['accuracy']:.3f} "
            f"loss={info['loss']:.4f} steps={info['steps']}"
        )


if __name__ == "__main__":
    main()
