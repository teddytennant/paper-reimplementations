"""Print one closed-form vs RK4 error and a 20-step smoke-train loss drop.

Run from the repo root with PYTHONPATH=src:

    python -m ananke.demo
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ananke.block import train_tiny_classifier
from ananke.flow import flow_vs_rk4_error


def flow_rk4_error():
    """Max-abs error of Phi_t against polar RK4 at dt=1e-3, t=1, float64."""
    jax.config.update("jax_enable_x64", True)
    x = jnp.array([1.7, -0.4], dtype=jnp.float64)
    c = jnp.array([0.2, 0.1], dtype=jnp.float64)
    err = flow_vs_rk4_error(
        x,
        c,
        R=jnp.asarray(1.0, dtype=jnp.float64),
        beta=jnp.asarray(0.8, dtype=jnp.float64),
        alpha_eff=jnp.asarray(1.2, dtype=jnp.float64),
        a=jnp.asarray(0.5, dtype=jnp.float64),
        t=1.0,
        dt=1e-3,
    )
    return err


def main():
    err = flow_rk4_error()
    loss0, loss20 = train_tiny_classifier(steps=20, seed=0)
    print(f"flow error vs RK4 (dt=1e-3, t=1): {float(err):.6e}")
    print(
        f"train loss drop: {float(loss0):.6f} -> {float(loss20):.6f} "
        f"(delta {float(loss0 - loss20):.6f})"
    )


if __name__ == "__main__":
    main()
