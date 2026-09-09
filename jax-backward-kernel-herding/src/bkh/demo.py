"""Tiny demo: Backward Kernel Herding on synthetic 2D points."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from bkh.bkh import backward_kernel_herding, kernelized_backward_herding
from bkh.kernels import gaussian_kernel, mmd2_features


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backward kernel herding demo")
    parser.add_argument("--n", type=int, default=64, help="number of points")
    parser.add_argument("--keep", type=int, default=32, help="coreset size")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args(argv)

    key = jax.random.PRNGKey(args.seed)
    x = jax.random.normal(key, (args.n, 2), dtype=jnp.float64)
    remaining, removed, eps, mu = backward_kernel_herding(x, args.keep)
    k = gaussian_kernel(x, x, sigma=1.0)
    rem_k, rem_order = kernelized_backward_herding(k, args.keep)
    mmd = mmd2_features(x[remaining], x)
    print(
        f"n={args.n} keep={args.keep} removed={int(removed.size)} "
        f"remaining_mmd2={float(mmd):.6e} "
        f"kernelized_first_removed={int(rem_order[0])}"
    )
    return remaining, rem_k, eps, mu


if __name__ == "__main__":
    main()
