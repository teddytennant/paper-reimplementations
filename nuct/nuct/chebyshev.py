"""Equispaced Chebyshev matrix and the NUCT identity (Definition 1.1, Theorem 5.1).

Index convention used everywhere in this package: a transform matrix has
**rows = frequency / degree** and **columns = nodes**. Theorem 5.1 writes
``(F_t)_{jk}`` with that order. Section 4 renames the same matrix
``(F_t)_{kj}``. The entries are the same.
"""

from __future__ import annotations

from nuct._config import ensure_x64

ensure_x64()

import jax.numpy as jnp


def nuct_order(N: int) -> int:
    """Return ``q`` with ``N = 2^q`` and ``q >= 2``.

    Definition 1.1 requires a power-of-two order at least 4.
    """
    N = int(N)
    if N < 4 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of two and at least 4, got {N}")
    return N.bit_length() - 1


def equispaced_nodes(N: int) -> jnp.ndarray:
    """Nodes ``x_k = -1 + 2k/N`` for ``k = 0, ..., N-1`` (Definition 1.1).

    For ``N = 2^q`` these are exact dyadics in float64 as long as ``q <= 53``.
    """
    nuct_order(N)
    k = jnp.arange(N, dtype=jnp.float64)
    return -1.0 + 2.0 * k / jnp.asarray(N, dtype=jnp.float64)


def chebyshev_T(degree, x) -> jnp.ndarray:
    """``T_n(x) = cos(n arccos x)``, the paper's definition.

    ``degree`` and ``x`` broadcast. Values outside ``[-1, 1]`` by a float
    ulp are clipped so ``arccos`` stays real; exact endpoints are unchanged.
    """
    degree = jnp.asarray(degree)
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.cos(degree * jnp.arccos(jnp.clip(x, -1.0, 1.0)))


def nuct_times(x) -> jnp.ndarray:
    """``t = arccos(x) / (2π)``. On Definition 1.1 nodes this lies in ``(0, 1/2]``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return jnp.arccos(jnp.clip(x, -1.0, 1.0)) / (2.0 * jnp.pi)


def reflected_times(t) -> jnp.ndarray:
    """``t^- = (-t) mod 1``, the negative nodes of Theorem 5.1."""
    t = jnp.asarray(t, dtype=jnp.float64)
    return jnp.mod(-t, 1.0)


def chebyshev_matrix(N: int) -> jnp.ndarray:
    """Normalized matrix ``(C_N)_{jk} = T_j(x_k) / sqrt(N)`` (Eq. (3)).

    Shape ``(N, N)``, float64, row ``j`` = degree, column ``k`` = node.
    """
    nuct_order(N)
    x = equispaced_nodes(N)
    degree = jnp.arange(N)[:, None]
    return chebyshev_T(degree, x[None, :]) / jnp.sqrt(jnp.asarray(N, dtype=jnp.float64))


def unnormalized_coefficients(values) -> jnp.ndarray:
    """``c_j = sum_k f_k T_j(x_k)``, the unnormalized coefficients of Definition 1.1."""
    values = jnp.asarray(values, dtype=jnp.float64)
    N = int(values.shape[0])
    nuct_order(N)
    x = equispaced_nodes(N)
    degree = jnp.arange(N)[:, None]
    return chebyshev_T(degree, x[None, :]) @ values


def coefficients_from_matrix(values) -> jnp.ndarray:
    """``c = sqrt(N) C_N f`` (Eq. (3))."""
    values = jnp.asarray(values, dtype=jnp.float64)
    N = int(values.shape[0])
    scale = jnp.sqrt(jnp.asarray(N, dtype=jnp.float64))
    return scale * (chebyshev_matrix(N) @ values)


def nudft_matrix(times, n_freq: int | None = None) -> jnp.ndarray:
    """Unitary nonuniform DFT, ``(F_t)_{jk} = N^{-1/2} exp(-2π i j t_k)``.

    ``times`` has one entry per column. ``N`` defaults to the number of
    columns (the square case of Theorem 5.1 and Eq. (31)).
    """
    times = jnp.asarray(times, dtype=jnp.float64)
    n_nodes = int(times.shape[0])
    N = n_nodes if n_freq is None else int(n_freq)
    if N < 1:
        raise ValueError("n_freq must be positive")
    frequency = jnp.arange(N, dtype=jnp.float64)[:, None]
    scale = jnp.sqrt(jnp.asarray(N, dtype=jnp.float64))
    return jnp.exp(-2j * jnp.pi * frequency * times[None, :]) / scale


def unitary_dft(N: int) -> jnp.ndarray:
    """Unitary DFT ``F_{kσ} = N^{-1/2} exp(-2π i k σ / N)``.

    Same sign and normalization as ``nudft_matrix`` on the uniform grid
    ``σ/N``. This is the matrix in Eq. (31), not the unnormalized FFT.
    """
    if int(N) < 1:
        raise ValueError("N must be positive")
    grid = jnp.arange(N, dtype=jnp.float64) / jnp.asarray(N, dtype=jnp.float64)
    return nudft_matrix(grid, n_freq=int(N))


def theorem_51_nodes(N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact nodes ``(t, t^-)`` of Theorem 5.1.

    ``t_0 = 1/2`` is written exactly. Float64 ``arccos(-1)/(2π)`` is already
    ``1/2``, but the endpoint is the one node the paper treats as exact, so
    it is not left to a transcendental evaluation.
    """
    nuct_order(N)
    t = nuct_times(equispaced_nodes(N)).at[0].set(0.5)
    return t, reflected_times(t)


def nuct_from_nufft(N: int) -> jnp.ndarray:
    """``(F_t + F_{t^-}) / 2`` with the exact nodes of Theorem 5.1."""
    t, t_minus = theorem_51_nodes(N)
    return 0.5 * (nudft_matrix(t) + nudft_matrix(t_minus))
