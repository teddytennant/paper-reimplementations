"""Fourier projection reconstruction of latent curves (paper eqs. 3–4)."""

from __future__ import annotations

import jax.numpy as jnp

from flr_noisy.basis import fourier_basis_matrix


def fourier_coefficients(Z: jnp.ndarray, n_basis: int) -> jnp.ndarray:
    """x̃_{i,j} = (1/p) ∑_h Z_i(t_h) φ_j(t_h).

    Z has shape (n, p) or (p,). Returns (n, n_basis) or (n_basis,).
    """
    Z = jnp.asarray(Z, dtype=jnp.float64)
    squeeze = Z.ndim == 1
    if squeeze:
        Z = Z[None, :]
    p = Z.shape[-1]
    phi = fourier_basis_matrix(p, n_basis)
    coeffs = (Z @ phi.T) / p
    if squeeze:
        return coeffs[0]
    return coeffs


def reconstruct_on_grid(coeffs: jnp.ndarray, p: int) -> jnp.ndarray:
    """X̃_i(t_h) = ∑_j x̃_{i,j} φ_j(t_h)."""
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    squeeze = coeffs.ndim == 1
    if squeeze:
        coeffs = coeffs[None, :]
    phi = fourier_basis_matrix(p, int(coeffs.shape[-1]))
    grid = coeffs @ phi
    if squeeze:
        return grid[0]
    return grid


def reconstruct(Z: jnp.ndarray, n_basis: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (coefficients, reconstructed curves on the observation grid)."""
    Z = jnp.asarray(Z, dtype=jnp.float64)
    p = Z.shape[-1]
    coeffs = fourier_coefficients(Z, n_basis)
    return coeffs, reconstruct_on_grid(coeffs, p)
