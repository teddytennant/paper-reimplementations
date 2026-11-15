"""Random cosine features and kernel spectral sampling (arXiv:2609.16406)."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def cosine_features(x: Array, omega: Array, b: Array) -> Array:
    """Random cosine map phi(x, omega, b) = cos(<omega, x> + b).

    Parameters
    ----------
    x : (..., d)
    omega : (n, d)
    b : (n,)

    Returns
    -------
    phi : (..., n)
    """
    x = jnp.asarray(x)
    if x.ndim == 1 and omega.shape[-1] == 1:
        x = x[:, None]
    return jnp.cos(x @ omega.T + b)


def sample_bias(key: Array, n: int) -> Array:
    """b ~ Uniform[0, 2 pi]."""
    return jax.random.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)


def sample_omega(
    key: Array,
    n: int,
    dim: int,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    nu: float = 2.5,
) -> Array:
    """Sample hidden frequencies from a kernel spectral density.

    Gaussian: omega ~ N(0, 1/sigma^4 I).
    Matern: independent Student-t with df=2 nu, location 0, scale 1/(2 pi sigma).
    Laplace: tensor-product Cauchy with scale 1/sigma.
    """
    kernel = kernel.lower()
    if kernel in ("gaussian", "rbf"):
        std = 1.0 / (sigma ** 2)
        return jax.random.normal(key, (n, dim)) * std
    if kernel in ("matern", "matérn"):
        df = 2.0 * nu
        scale = 1.0 / (2.0 * jnp.pi * sigma)
        key_z, key_v = jax.random.split(key)
        z = jax.random.normal(key_z, (n, dim))
        chi2 = 2.0 * jax.random.gamma(key_v, df / 2.0, (n, dim))
        t = z / jnp.sqrt(chi2 / df)
        return scale * t
    if kernel in ("laplace", "cauchy"):
        scale = 1.0 / sigma
        return scale * jax.random.cauchy(key, (n, dim))
    raise ValueError(f"unknown kernel {kernel!r}")


def init_random_features(
    key: Array,
    n_features: int,
    dim: int,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    nu: float = 2.5,
    init_c: str = "zeros",
) -> dict:
    """Sample frozen (omega, b) and a trainable readout c of shape (n_features,)."""
    k_w, k_b, k_c = jax.random.split(key, 3)
    omega = sample_omega(k_w, n_features, dim, kernel=kernel, sigma=sigma, nu=nu)
    b = sample_bias(k_b, n_features)
    if init_c == "zeros":
        c = jnp.zeros((n_features,))
    else:
        c = 0.01 * jax.random.normal(k_c, (n_features,))
    return {"omega": omega, "b": b, "c": c}


def product_features(
    x: Array,
    t: Array,
    omega_x: Array,
    b_x: Array,
    omega_t: Array,
    b_t: Array,
) -> Array:
    """Kronecker product features phi(x,t) = phi_x(x) ⊗ phi_t(t).

    Parameters
    ----------
    x : (n, d_x) or (n,)
    t : (n, d_t) or (n,)

    Returns
    -------
    phi : (n, n_x * n_t)
    """
    phi_x = cosine_features(x, omega_x, b_x)
    phi_t = cosine_features(t, omega_t, b_t)
    return (phi_x[:, :, None] * phi_t[:, None, :]).reshape(phi_x.shape[0], -1)


def init_product_features(
    key: Array,
    n_x: int,
    n_t: int,
    dim_x: int = 1,
    dim_t: int = 1,
    kernel_x: str = "gaussian",
    kernel_t: str = "gaussian",
    sigma_x: float = 1.0,
    sigma_t: float = 1.0,
    nu_x: float = 2.5,
    nu_t: float = 2.5,
    init_c: str = "zeros",
) -> dict:
    """Independent spatial/temporal features; readout length n_x * n_t."""
    k_wx, k_bx, k_wt, k_bt, k_c = jax.random.split(key, 5)
    omega_x = sample_omega(k_wx, n_x, dim_x, kernel=kernel_x, sigma=sigma_x, nu=nu_x)
    b_x = sample_bias(k_bx, n_x)
    omega_t = sample_omega(k_wt, n_t, dim_t, kernel=kernel_t, sigma=sigma_t, nu=nu_t)
    b_t = sample_bias(k_bt, n_t)
    n = n_x * n_t
    if init_c == "zeros":
        c = jnp.zeros((n,))
    else:
        c = 0.01 * jax.random.normal(k_c, (n,))
    return {
        "omega_x": omega_x,
        "b_x": b_x,
        "omega_t": omega_t,
        "b_t": b_t,
        "c": c,
    }


def split_frozen(params: dict) -> Tuple[dict, Array]:
    """Return frozen feature dict and trainable c."""
    c = params["c"]
    frozen = {k: v for k, v in params.items() if k != "c"}
    return frozen, c
