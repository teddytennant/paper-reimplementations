"""Draw ~256 samples from a 2D standard Gaussian via SPHMC."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sphmc.sampler import SPHMCConfig, sample_quadratic


def main() -> None:
    key = jax.random.PRNGKey(0)
    A = jnp.eye(2)
    config = SPHMCConfig(h=0.1, n_picard=1, J=4, eta=0.05, gamma=1.0)
    samples = sample_quadratic(
        key,
        A,
        config,
        n_steps=40,
        n_samples=256,
        recover_rgo=True,
    )
    mean = jnp.mean(samples, axis=0)
    var = jnp.var(samples, axis=0)
    print(f"n={samples.shape[0]} d={samples.shape[1]}")
    print("mean =", jnp.asarray(mean))
    print("var  =", jnp.asarray(var))


if __name__ == "__main__":
    main()
