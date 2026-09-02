"""A short demo: the Figure 2 sweep, then one pass of Algorithm 1."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .convergence import (
    figure2_setup,
    reverse_attention_trajectory,
    sigma_schedule,
    weight_entropy,
)
from .generate import generate


def main():
    data, y, mean = figure2_setup()
    sigmas = sigma_schedule()
    traj = reverse_attention_trajectory(y, data, sigmas)
    entropy = weight_entropy(traj.weights)
    to_mean = jnp.linalg.norm(traj.points - mean, axis=-1)
    final = traj.points[-1]
    nearest = int(jnp.argmin(jnp.sum((data - final) ** 2, axis=-1)))

    print("Figure 2, Sec. II-D")
    print(f"  sigma {float(sigmas[0]):.3f} -> {float(sigmas[-1]):.3f} in {len(sigmas)} steps")
    print(f"  closest approach to the dataset mean: {float(jnp.min(to_mean)):.4f}")
    print(f"  weight entropy: {float(entropy[0]):.4f} nats -> {float(entropy[-1]):.4f}")
    print(f"  ends {float(jnp.linalg.norm(final - data[nearest])):.4f} from s{nearest + 1}")

    key = jax.random.key(0)
    kd, kq, kn = jax.random.split(key, 3)
    latents = jax.random.normal(kd, (256, 3, 7, 7))
    out = generate(latents, latents[7], k=3, sigma=0.01, eta=0.01, key=kn)

    print("Algorithm 1, Sec. V")
    print(f"  neighbours: {out.neighbor_indices.tolist()}")
    print(f"  mean latent radius r = {float(out.radius):.4f}")
    print(f"  ||z|| after line 8 = {float(jnp.linalg.norm(out.rescaled_latent)):.4f}")
    print(f"  weights sum over neighbours = {float(jnp.mean(out.weights.sum(0))):.6f}")


if __name__ == "__main__":
    main()
