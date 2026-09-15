"""Synthetic covariates matching the paper's simulation sketch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from flr_noisy.basis import fourier_basis_at, fourier_basis_matrix


@dataclass
class SyntheticData:
    Z: np.ndarray
    y: np.ndarray
    X: np.ndarray
    xi: np.ndarray
    beta_coeffs: np.ndarray
    inner: np.ndarray
    lam: np.ndarray
    t: np.ndarray
    sigma2: float
    tau2: float


def kl_eigenvalues(n_basis: int, decay: float = 4.0) -> jnp.ndarray:
    """λ_j = j^{-decay}. Paper simulations use decay=4 (λ_j = 1/j^4)."""
    j = jnp.arange(1, n_basis + 1, dtype=jnp.float64)
    return j ** (-decay)


def decaying_beta_coeffs(n_basis: int, decay: float = 2.0, scale: float = 1.0) -> jnp.ndarray:
    """Slope coefficients b_j = scale / j^{decay} in the Fourier basis."""
    j = jnp.arange(1, n_basis + 1, dtype=jnp.float64)
    return scale * j ** (-decay)


def quadratic_beta_coeffs(n_basis: int, p_sim: int = 512) -> jnp.ndarray:
    """Fourier coefficients of β_1(t)=t(t-1) by Riemann sum on a fine grid."""
    t = jnp.linspace(0.0, 1.0, p_sim, endpoint=False)
    beta = t * (t - 1.0)
    phi = fourier_basis_at(t, n_basis)
    return (beta[:, None] * phi).mean(axis=0)


def generate_kl_covariates(
    key: jax.Array,
    n: int,
    p: int,
    n_kl: int,
    decay: float = 4.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """X_i(t) = ∑_{j=1}^{n_kl} ξ_{ij} φ_j(t), ξ_{ij} ~ N(0, λ_j)."""
    lam = kl_eigenvalues(n_kl, decay=decay)
    xi = jax.random.normal(key, (n, n_kl), dtype=jnp.float64) * jnp.sqrt(lam)
    phi = fourier_basis_matrix(p, n_kl)
    X = xi @ phi
    return X, xi, lam


def generate_synthetic(
    key: jax.Array,
    n: int,
    p: int,
    *,
    n_kl: int = 33,
    eigenvalue_decay: float = 4.0,
    beta: str = "fourier",
    beta_decay: float = 2.0,
    beta_dim: int = 9,
    sigma2: float = 0.1,
    tau2: float = 0.1,
) -> SyntheticData:
    """Noisy regular-grid FLR sample.

    beta='fourier' uses decaying Fourier coefficients (exact inner products).
    beta='quadratic' uses β(t)=t(t-1) with Riemann inner products on the p-grid.
    """
    key_x, key_eps, key_eta = jax.random.split(key, 3)
    X, xi, lam = generate_kl_covariates(
        key_x, n, p, n_kl, decay=eigenvalue_decay
    )
    if beta == "fourier":
        bdim = min(beta_dim, n_kl)
        b = jnp.zeros(n_kl, dtype=jnp.float64)
        b = b.at[:bdim].set(decaying_beta_coeffs(bdim, decay=beta_decay))
        inner = (xi * b).sum(axis=1)
    elif beta == "quadratic":
        t = jnp.arange(p, dtype=jnp.float64) / p
        b_fn = t * (t - 1.0)
        inner = (X * b_fn).mean(axis=1)
        b = quadratic_beta_coeffs(n_kl)
    elif beta == "one_harmonic":
        b = jnp.zeros(n_kl, dtype=jnp.float64)
        b = b.at[1].set(1.0)
        inner = (xi * b).sum(axis=1)
    else:
        raise ValueError(f"unknown beta={beta!r}")

    eps = jax.random.normal(key_eps, (n,), dtype=jnp.float64) * math.sqrt(sigma2)
    eta = jax.random.normal(key_eta, (n, p), dtype=jnp.float64) * math.sqrt(tau2)
    y = inner + eps
    Z = X + eta
    return SyntheticData(
        Z=np.asarray(Z, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        X=np.asarray(X, dtype=np.float64),
        xi=np.asarray(xi, dtype=np.float64),
        beta_coeffs=np.asarray(b, dtype=np.float64),
        inner=np.asarray(inner, dtype=np.float64),
        lam=np.asarray(lam, dtype=np.float64),
        t=np.arange(p, dtype=np.float64) / p,
        sigma2=float(sigma2),
        tau2=float(tau2),
    )
