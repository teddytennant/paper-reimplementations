"""A few local DKRGD steps on a 1D noisy sine (CPU)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dkrgd import fit_dkrgd, predict_dkrgd, step_size_bound


def main() -> None:
    key = jax.random.PRNGKey(0)
    n = 40
    x = jnp.linspace(-2.0, 2.0, n)
    X = x[:, None]
    noise = 0.15 * jax.random.normal(key, (n,))
    y = jnp.sin(2.0 * x) + noise

    eta = 0.5 * step_size_bound(kappa=1.0)
    locals_ = fit_dkrgd(
        X,
        y,
        m=2,
        n_steps=8,
        eta=eta,
        sigma=1.0,
        loss="welsch",
        lengthscale=0.5,
    )
    pred = predict_dkrgd(locals_, X)
    mse = float(jnp.mean((pred - y) ** 2))
    print(f"machines={len(locals_)} n_steps=8 mse={mse:.4f}")
    print("pred[:5] =", jnp.asarray(pred[:5]))


if __name__ == "__main__":
    main()
