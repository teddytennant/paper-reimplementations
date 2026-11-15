"""Tiny CPU smoke of one GenAIMMD cycle (not the paper's 50 x 200-epoch run)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from genaimmd.loop import init_state, run_cycle


def main():
    key = jax.random.PRNGKey(0)
    state = init_state(key, n_a=8, n_b=8, n_path=8)
    key, k = jax.random.split(key)
    out = run_cycle(
        state,
        k,
        n_committor_steps=5,
        n_bg_steps=5,
        n_relax=10,
        max_shoot_steps=30,
        batch_size=16,
    )
    print("committor_loss", float(out["committor_loss"]))
    print("bg_loss", float(out["bg_loss"]))
    print("n_samples", int(out["samples"].shape[0]))


if __name__ == "__main__":
    main()
