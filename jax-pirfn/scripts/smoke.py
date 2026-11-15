"""Train 1D Poisson PIRFN for a few hundred Adam steps and print relative L2."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from jax_pirfn.features import init_random_features
from jax_pirfn.losses import relative_l2
from jax_pirfn.pde import apply_1d, make_poisson_1d_loss, poisson_1d_source
from jax_pirfn.train import train_readout


def main() -> int:
    key = jax.random.PRNGKey(3)
    params = init_random_features(
        key, n_features=96, dim=1, kernel="gaussian", sigma=0.45
    )
    x_int = jnp.concatenate(
        [jnp.linspace(0.0, 1.0, 80)[1:-1], jnp.array([0.01, 0.99])]
    )
    _, f = poisson_1d_source(x_int)
    x_bd = jnp.array([0.0, 1.0])
    g_bd = jnp.zeros((2,))
    loss_fn = make_poisson_1d_loss(
        params["omega"], params["b"], x_int, f, x_bd, g_bd, lam_l2=1e-6
    )
    c, last_loss = train_readout(params["c"], loss_fn, n_steps=400, lr=2e-2)
    x_test = jnp.linspace(0.0, 1.0, 200)
    u_true, _ = poisson_1d_source(x_test)
    u_hat = apply_1d(x_test, params["omega"], params["b"], c)
    err = float(relative_l2(u_hat, u_true))
    print(f"relative_L2={err:.6f} last_loss={float(last_loss):.6e}")
    if err < 0.15:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
