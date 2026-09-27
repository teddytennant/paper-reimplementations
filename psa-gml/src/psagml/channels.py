"""i.i.d. Rayleigh (CN(0,1)) channels for the PSA-GML kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def complex_normal(key, shape, dtype=jnp.complex64):
    """i.i.d. CN(0,1) array: real/imag N(0, 1/2)."""
    real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
    k1, k2 = jax.random.split(key)
    real = jax.random.normal(k1, shape, dtype=real_dtype)
    imag = jax.random.normal(k2, shape, dtype=real_dtype)
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype))
    return (real + 1j * imag) / scale


def rayleigh_channels(key, Nt, N, Kt, Kr, dtype=jnp.complex64):
    """Draw G (N x Nt), Ht (Kt x N), Hr (Kr x N) i.i.d. CN(0,1)."""
    k1, k2, k3 = jax.random.split(key, 3)
    G = complex_normal(k1, (N, Nt), dtype)
    Ht = complex_normal(k2, (Kt, N), dtype)
    Hr = complex_normal(k3, (Kr, N), dtype)
    return G, Ht, Hr
