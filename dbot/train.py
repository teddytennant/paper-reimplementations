"""Fit a 4-state softmax kernel to a drifted random walk."""

import sys

import jax
import numpy as np

from dbot import fit_kernel


def drifted_walk(n=4):
    """P(i, i) = 0.5 and P(i, i+1 mod n) = 0.5. Rows already sum to 1."""
    kernel = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        kernel[i, i] = 0.5
        kernel[i, (i + 1) % n] = 0.5
    return kernel


def main():
    n = 4
    target = drifted_walk(n)
    cost = 1.0 - np.eye(n)
    result = fit_kernel(
        target,
        cost=cost,
        gamma=0.9,
        x0=0,
        y0=0,
        steps=60,
        lr=1.0,
        key=jax.random.PRNGKey(0),
    )
    initial = float(result["distance_trace"][0])
    final = float(result["distance_trace"][-1])
    print(f"initial distance {initial:.6f}")
    print(f"final distance {final:.6f}")
    return 0 if final < initial else 1


if __name__ == "__main__":
    sys.exit(main())
