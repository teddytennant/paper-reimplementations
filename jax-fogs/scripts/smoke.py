import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from jax_fogs import alpha_grid, fogs, simulate_folti


def main():
    A = jnp.array([[0.25, 0.10], [-0.05, 0.20]])
    alpha = jnp.array([0.2, 0.4])
    x0 = jnp.array([1.0, -0.7])
    traj = simulate_folti(A, alpha, x0, jnp.zeros((40, 2)))
    a_hat, A_hat = fogs(traj, alpha_grid(0.05, 0.55, 11))
    print("alpha_true", alpha)
    print("alpha_hat ", a_hat)
    print("A_true")
    print(A)
    print("A_hat")
    print(A_hat)


if __name__ == "__main__":
    main()
