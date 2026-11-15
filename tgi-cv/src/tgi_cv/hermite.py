"""Probabilists Hermite polynomials, orthonormal under N(0, I)."""

from __future__ import annotations

import math

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

import jax.numpy as jnp


def hermite_probabilists(n: int, x):
    """He_0 = 1, He_1 = x, He_{n+1} = x*He_n - n*He_{n-1}."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    x = jnp.asarray(x)
    if n == 0:
        return jnp.ones_like(x)
    h_prev = jnp.ones_like(x)
    h = x
    if n == 1:
        return h
    for k in range(1, n):
        h_prev, h = h, x * h - k * h_prev
    return h


def hermite_multivariate(k, z):
    """H_k(z) = prod_j He_{k_j}(z_j) / sqrt(k_j!)."""
    z = jnp.asarray(z)
    value = jnp.ones(z.shape[:-1], dtype=z.dtype)
    for j, kj in enumerate(tuple(int(v) for v in k)):
        he = hermite_probabilists(kj, z[..., j])
        value = value * he / jnp.sqrt(math.factorial(kj))
    return value


def evaluate_modes(modes, z):
    """Stack H_k(z) for each multi-index. The last axis indexes modes."""
    z = jnp.asarray(z)
    modes = [tuple(int(v) for v in k) for k in modes]
    if not modes:
        return jnp.zeros(z.shape[:-1] + (0,), dtype=z.dtype)
    return jnp.stack([hermite_multivariate(k, z) for k in modes], axis=-1)


def mode_order(k) -> int:
    return int(sum(int(v) for v in k))
