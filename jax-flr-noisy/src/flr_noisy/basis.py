"""Trigonometric Fourier basis on the regular grid t_h = h/p."""

from __future__ import annotations

import math

import jax.numpy as jnp

KAPPA = 4.01
ALPHA = 5.0


def dimension(m: int) -> int:
    """D_m = 2m + 1, dimension of S_m."""
    if m < 1:
        raise ValueError("model index m must be >= 1")
    return 2 * m + 1


def model_index(n_basis: int) -> int:
    """Largest m with D_m <= n_basis."""
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    return max((n_basis - 1) // 2, 1)


def regular_grid(p: int) -> jnp.ndarray:
    """Observation grid t_h = h/p for h = 0, ..., p-1."""
    return jnp.arange(p, dtype=jnp.float64) / p


def fourier_basis_matrix(p: int, n_basis: int) -> jnp.ndarray:
    """Evaluate φ_1, ..., φ_{n_basis} on the regular p-grid.

    φ_1 = 1, φ_{2k} = √2 cos(2π k t), φ_{2k+1} = √2 sin(2π k t).
    Returns shape (n_basis, p).
    """
    if n_basis < 1:
        raise ValueError("n_basis must be >= 1")
    t = regular_grid(p)
    j = jnp.arange(n_basis, dtype=jnp.int32)
    k = (j + 1) // 2
    is_const = j == 0
    is_cos = (j % 2) == 1
    arg = 2.0 * jnp.pi * k[:, None].astype(jnp.float64) * t[None, :]
    cos_part = math.sqrt(2.0) * jnp.cos(arg)
    sin_part = math.sqrt(2.0) * jnp.sin(arg)
    const = jnp.ones((n_basis, p), dtype=jnp.float64)
    return jnp.where(
        is_const[:, None],
        const,
        jnp.where(is_cos[:, None], cos_part, sin_part),
    )


def fourier_basis_at(t: jnp.ndarray, n_basis: int) -> jnp.ndarray:
    """Evaluate φ_1, ..., φ_{n_basis} at times t. Shape t.shape + (n_basis,)."""
    t = jnp.asarray(t, dtype=jnp.float64)
    j = jnp.arange(n_basis, dtype=jnp.int32)
    k = (j + 1) // 2
    is_const = j == 0
    is_cos = (j % 2) == 1
    arg = 2.0 * jnp.pi * k.astype(jnp.float64) * t[..., None]
    vals = jnp.where(
        is_const,
        1.0,
        jnp.where(
            is_cos,
            math.sqrt(2.0) * jnp.cos(arg),
            math.sqrt(2.0) * jnp.sin(arg),
        ),
    )
    return vals


def evaluate_fourier(t: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    """∑_j α_j φ_j(t) for 1d coefficient vector α."""
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    basis = fourier_basis_at(t, int(coeffs.shape[-1]))
    return jnp.sum(basis * coeffs, axis=-1)


def discrete_inner_product(f_grid: jnp.ndarray, g_grid: jnp.ndarray) -> jnp.ndarray:
    """⟨f, g⟩_p = (1/p) ∑_h f(t_h) g(t_h)."""
    f_grid = jnp.asarray(f_grid, dtype=jnp.float64)
    g_grid = jnp.asarray(g_grid, dtype=jnp.float64)
    p = f_grid.shape[-1]
    return jnp.sum(f_grid * g_grid, axis=-1) / p


def lemma_a1_inner_product(l: int, j: int, p: int) -> float:
    """Closed form of ⟨φ_l, φ_j⟩_p from Lemma A1 (1-based indices)."""
    if l < 1 or j < 1:
        raise ValueError("basis indices are 1-based")
    two_p = 2 * p
    if l == 1 and j == 1:
        return 1.0
    if l % 2 == 0 and j % 2 == 0:
        return float((l - j) % two_p == 0) + float((l + j) % two_p == 0)
    if l % 2 == 1 and j % 2 == 1 and l > 1 and j > 1:
        return float((l - j) % two_p == 0) - float((l + j - 2) % two_p == 0)
    if j == 1 and l % 2 == 0:
        return math.sqrt(2.0) if l % two_p == 0 else 0.0
    if l == 1 and j % 2 == 0:
        return math.sqrt(2.0) if j % two_p == 0 else 0.0
    return 0.0


def theoretical_recon_dim(n: int, p: int) -> int:
    """Largest odd D_{N_{n,p}} with D < min(n / ln² n, p). At least 1."""
    if n < 3:
        cap = float(p)
    else:
        cap = min(n / (math.log(n) ** 2), float(p))
    d = int(math.floor(cap - 1e-15))
    if d % 2 == 0:
        d -= 1
    d = max(d, 1)
    # S_1 has dimension 3; clip up so the collection M is non-empty when p allows it.
    if p > 3:
        d = max(d, 3)
    if d >= p:
        d = p - 1 if (p - 1) % 2 == 1 else p - 2
        d = max(d, 1)
    return d


def theoretical_max_model_index(n: int, p: int) -> int:
    """N_{n,p} = largest m with D_m < min(n / ln² n, p). At least 1."""
    return model_index(theoretical_recon_dim(n, p))


def eigenvalue_threshold(n: int, alpha: float = ALPHA) -> float:
    """s_n = 2 n^{-α} (1 - 1/√ln n)."""
    if n < 3:
        return 0.0
    return (2.0 / (n ** alpha)) * (1.0 - 1.0 / math.sqrt(math.log(n)))


def penalty(m: int, n: int, sigma2: float, kappa: float = KAPPA) -> float:
    """pen(m) = κ D_m σ² / n."""
    return kappa * dimension(m) * sigma2 / n
