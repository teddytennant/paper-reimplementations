"""Print the max absolute FlashBoB error against autodiff on one random problem."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flashbob.bob import flashbob_autodiff, flashbob_tiled


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    n, d, br, bc = 15, 7, 4, 6
    keys = jax.random.split(key, 7)
    Q, K, V, dO, UQ, UK, UV = [
        jax.random.normal(k, (n, d), dtype=jnp.float64) for k in keys
    ]
    tiled = flashbob_tiled(Q, K, V, dO, UQ, UK, UV, br, bc, causal=True)
    grad = flashbob_autodiff(Q, K, V, dO, UQ, UK, UV, causal=True)
    err = max(float(jnp.max(jnp.abs(tiled[i] - grad[i]))) for i in range(4))
    print(f"max abs error vs autodiff: {err:.3e}")


if __name__ == "__main__":
    main()
