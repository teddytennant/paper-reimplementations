"""Tiny LM batch: init PRM transformer, 20 AdamW steps, print losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_prm import PRMConfig, adamw_init, adamw_step, init, loss_fn


def main():
    cfg = PRMConfig(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=32,
        context=8,
        n_slots=4,
        d_mem=16,
        dropout=0.0,
    )
    key = jax.random.PRNGKey(0)
    k_init, k_data = jax.random.split(key)
    params = init(k_init, cfg)
    tokens = jax.random.randint(k_data, (4, 8), 0, cfg.vocab_size)

    loss0 = float(loss_fn(params, tokens, cfg))
    print(f"train_loss_0={loss0:.6f}")

    opt = adamw_init(params)

    @jax.jit
    def step(params, opt):
        return adamw_step(params, opt, tokens, cfg)

    p = params
    for _ in range(20):
        p, opt, _ = step(p, opt)

    loss20 = float(loss_fn(p, tokens, cfg))
    print(f"train_loss_20={loss20:.6f}")
    if not (jnp.isfinite(loss0) and jnp.isfinite(loss20)):
        raise SystemExit("non-finite loss")


if __name__ == "__main__":
    main()
