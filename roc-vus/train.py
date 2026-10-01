"""Simulate a two-class Gaussian mixture and print VUS estimates."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from roc_vus import cross_fit_vus


def main():
    key = jax.random.PRNGKey(0)
    key_x, key_split = jax.random.split(key)
    n = 200
    n_per = n // 2
    mu = jnp.array([-1.0, 1.0])
    # Equal-variance two-class VUS equals the class-conditional AUC, Phi(delta / sqrt(2)).
    population = float(jax.scipy.stats.norm.cdf((mu[1] - mu[0]) / jnp.sqrt(2.0)))

    labels_idx = jnp.concatenate(
        [
            jnp.zeros(n_per, dtype=jnp.int32),
            jnp.ones(n_per, dtype=jnp.int32),
        ]
    )
    labels_idx = jax.random.permutation(key_split, labels_idx)
    features = (mu[labels_idx] + jax.random.normal(key_x, (n,)))[:, None]
    labels = jax.nn.one_hot(labels_idx, 2, dtype=jnp.float64)

    detail = cross_fit_vus(
        features,
        labels,
        shrink=0.0,
        seed=1,
        n_steps=250,
        learning_rate=0.4,
        l2=1e-3,
        return_details=True,
    )
    print(f"population VUS: {population:.6f}")
    print(f"label-estimator VUS: {float(detail['label']):.6f}")
    print(f"plug-in VUS: {float(detail['plugin']):.6f}")


if __name__ == "__main__":
    main()
