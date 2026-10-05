"""One modestly larger draw of the direct test. Prints T_hat, q_hat, and the decision."""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax.numpy as jnp

from tgg import (
    band_edges,
    direct_test,
    generate_linear_process,
    tridiagonal_precision,
)


def main():
    n = 120
    p = 16
    kappa = 0.3
    beta = 0.9
    alpha = 0.05
    l = math.floor(n ** (1.0 / 3.0))
    precision = tridiagonal_precision(p, kappa)
    covariance = jnp.linalg.inv(precision)
    series = generate_linear_process(n, covariance, beta=beta, seed=0)
    result = direct_test(series, band_edges(p, 1), l=l, alpha=alpha)
    print(f"T_hat={result.statistic:.6f}")
    print(f"q_hat={result.critical_value:.6f}")
    print(f"reject={result.reject}")


if __name__ == "__main__":
    main()
