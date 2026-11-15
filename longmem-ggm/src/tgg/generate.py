"""Synthetic Gaussian linear processes with a power-law coefficient decay.

The truncated process in the appendix is

    X_i* = sum_{t=0}^{N-1} A_t eps_{i-t},

with N = n^2 in the simulation section. Coefficient matrices follow the
power-law bound in Condition A: A_t = a_t Sigma^{1/2}, where
a_t is proportional to (max(1, t))^{-beta} and sum_t a_t^2 = 1.
Innovations are i.i.d. N(0, I_p). The sum is a circular convolution,
evaluated by FFT and split into consecutive length-n blocks, which is the
computational scheme in the appendix.

The appendix writes a flattened circulant of length N*d and reads the first
coordinate of each d-block. That extraction does not equal the row-vector
product (A_t)_{j.} · eps_{i-t} when d > 1. This module applies the FFT
channel-wise, which is that product, and then colors the series by
Sigma^{1/2}. The circular process has marginal covariance exactly Sigma.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def tridiagonal_precision(p: int, kappa: float) -> jnp.ndarray:
    """Precision with 1 on the diagonal and kappa on |i - j| = 1.

    This is the scaled-down band used by the tests. kappa in {0.2, 0.4}
    keeps the matrix positive definite for moderate p.
    """

    if p < 1:
        raise ValueError("p must be positive")
    omega = jnp.eye(p, dtype=jnp.float64)
    if p == 1:
        return omega
    idx = jnp.arange(p - 1)
    omega = omega.at[idx, idx + 1].set(kappa)
    omega = omega.at[idx + 1, idx].set(kappa)
    return omega


def band_edges(p: int, bandwidth: int = 1):
    """Undirected edges with 1 <= |i - j| <= bandwidth, 0-based."""

    if bandwidth < 0:
        raise ValueError("bandwidth must be nonnegative")
    edges = []
    for i in range(p):
        for j in range(i + 1, min(p, i + bandwidth + 1)):
            edges.append((i, j))
    return edges


def simulation_precision(
    p: int,
    theta0: float = 0.25,
    rho: float = 0.5,
    bandwidth: int = 4,
    delta0: float = 0.10,
) -> jnp.ndarray:
    """Precision from the simulation section.

    (Omega)_{ij} = theta0 * rho^{|i-j|-1} for 1 <= |i-j| <= bandwidth,
    and (Omega)_{ii} = 1 + sum_{j != i} |(Omega)_{ij}| + delta0.
    """

    off = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            lag = abs(i - j)
            if 1 <= lag <= bandwidth:
                off[i, j] = theta0 * (rho ** (lag - 1))
    diag = 1.0 + np.sum(np.abs(off), axis=1) + delta0
    omega = off + np.diag(diag)
    return jnp.asarray(omega, dtype=jnp.float64)


def power_law_coefficients(N: int, beta: float) -> jnp.ndarray:
    """l2-normalized coefficients a_t proportional to (max(1, t))^{-beta}."""

    if N < 1:
        raise ValueError("N must be positive")
    lags = jnp.arange(N, dtype=jnp.float64)
    raw = jnp.where(lags == 0, 1.0, lags ** (-float(beta)))
    return raw / jnp.sqrt(jnp.sum(raw**2))


def sample_iid(n: int, Sigma, seed: int) -> jnp.ndarray:
    """n i.i.d. N(0, Sigma) rows. Marginal law of the linear process at beta = infinity."""

    Sigma = jnp.asarray(Sigma, dtype=jnp.float64)
    p = int(Sigma.shape[0])
    chol = jnp.linalg.cholesky(Sigma)
    key = jax.random.PRNGKey(int(seed))
    z = jax.random.normal(key, (n, p), dtype=jnp.float64)
    return z @ chol.T


def generate_linear_process(
    n: int,
    Sigma,
    beta: float,
    seed: int,
    truncation: int | None = None,
    block: int = 0,
) -> jnp.ndarray:
    """One length-n block of the FFT coefficient process.

    truncation defaults to n^2, the simulation-section choice. block selects
    which consecutive segment of the long circular convolution to return.
    beta = 2 is short memory; beta = 0.9 is long memory inside the paper's
    Gaussian-approximation regime (beta > 3/4).
    """

    Sigma = jnp.asarray(Sigma, dtype=jnp.float64)
    p = int(Sigma.shape[0])
    N = n * n if truncation is None else int(truncation)
    if N < n:
        raise ValueError("truncation must be at least n")
    n_blocks = N // n
    if block < 0 or block >= n_blocks:
        raise ValueError(f"block must lie in 0..{n_blocks - 1}")
    key = jax.random.PRNGKey(int(seed))
    eps = jax.random.normal(key, (N, p), dtype=jnp.float64)
    coeff = power_law_coefficients(N, beta)
    filtered = jnp.fft.ifft(jnp.fft.fft(coeff)[:, None] * jnp.fft.fft(eps, axis=0), axis=0).real
    chol = jnp.linalg.cholesky(Sigma)
    series = filtered @ chol.T
    start = block * n
    return series[start : start + n]
