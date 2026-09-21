"""Density-matrix helpers used by the MEFPIA kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def hermitize(a: Array) -> Array:
    """Return (A + A^dagger)/2 so Hamiltonians stay Hermitian."""
    return 0.5 * (a + jnp.conjugate(jnp.swapaxes(a, -1, -2)))


def is_hermitian(a: Array, atol: float = 1e-6) -> Array:
    return jnp.allclose(a, jnp.conjugate(jnp.swapaxes(a, -1, -2)), atol=atol)


def is_density(rho: Array, atol: float = 1e-5) -> Array:
    """Hermitian, unit trace, and positive semidefinite within atol."""
    rho_h = hermitize(rho)
    hermitian = jnp.allclose(rho, rho_h, atol=atol)
    trace_ok = jnp.isclose(jnp.trace(rho_h).real, 1.0, atol=atol)
    evals = jnp.linalg.eigvalsh(rho_h).real
    psd = jnp.all(evals >= -atol)
    return hermitian & trace_ok & psd


def random_density(key: jax.Array, dim: int, dtype=jnp.complex128) -> Array:
    """Random full-rank density matrix from a Ginibre Wishart draw."""
    key_r, key_i = jax.random.split(key)
    real_dtype = jnp.finfo(dtype).dtype
    a = jax.random.normal(key_r, (dim, dim), dtype=real_dtype).astype(dtype)
    a = a + 1j * jax.random.normal(key_i, (dim, dim), dtype=real_dtype).astype(dtype)
    rho = a @ jnp.conjugate(a.T)
    rho = hermitize(rho)
    return (rho / jnp.trace(rho).real.astype(dtype)).astype(dtype)


def von_neumann_entropy(rho: Array, clip: float = 1e-12) -> Array:
    """S(rho) = -Tr(rho log rho) via a Hermitian eigendecomposition."""
    evals = jnp.clip(jnp.linalg.eigvalsh(hermitize(rho)).real, clip, None)
    return -jnp.sum(evals * jnp.log(evals))
