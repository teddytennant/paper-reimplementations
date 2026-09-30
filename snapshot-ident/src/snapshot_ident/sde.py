"""Planted-curl Ornstein-Uhlenbeck geometry and independent Gaussian slices."""
from __future__ import annotations

from . import _host

_host.preload()

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)


def rotation_generator(d: int, m: int, omega: float) -> jax.Array:
    """Antisymmetric Omega supported on the first m coordinates (m even)."""
    omega = jnp.asarray(omega, dtype=jnp.float64)
    Omega = jnp.zeros((d, d), dtype=jnp.float64)
    n_planes = m // 2
    for p in range(n_planes):
        i = 2 * p
        j = 2 * p + 1
        Omega = Omega.at[i, j].set(omega)
        Omega = Omega.at[j, i].set(-omega)
    return Omega


def planted_curl_H(d: int, m: int, nu: float, omega: float) -> jax.Array:
    """Linear drift F(x) = H x with H = -nu I + Omega."""
    nu = jnp.asarray(nu, dtype=jnp.float64)
    return -nu * jnp.eye(d, dtype=jnp.float64) + rotation_generator(d, m, omega)


def _diffusion_scale(nu: jax.Array, sigma2: jax.Array, t: jax.Array) -> jax.Array:
    """(sigma^2 / nu) (1 - exp(-2 nu t)), with the nu -> 0 limit 2 sigma^2 t."""
    two_nu_t = 2.0 * nu * t
    return jnp.where(
        jnp.abs(nu) < 1e-12,
        2.0 * sigma2 * t,
        (sigma2 / nu) * (1.0 - jnp.exp(-two_nu_t)),
    )


def ou_mean(mu0: jax.Array, t: float | jax.Array, H: jax.Array) -> jax.Array:
    """Mean of the linear SDE dZ = H Z dt + ... starting at mu0."""
    t = jnp.asarray(t, dtype=jnp.float64)
    return expm(t * H) @ mu0


def ou_covariance(
    Sigma0: jax.Array,
    t: float | jax.Array,
    nu: float | jax.Array,
    sigma2: float | jax.Array,
    Omega: jax.Array,
) -> jax.Array:
    """Centered covariance of planted-curl OU at time t.

    For H = -nu I + Omega with Omega^T = -Omega,
    Sigma_t = e^{-2 nu t} R(t) Sigma0 R(t)^T + (sigma^2 / nu) (1 - e^{-2 nu t}) I
    where R(t) = exp(t Omega).
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    d = Sigma0.shape[0]
    R = expm(t * Omega)
    decay = jnp.exp(-2.0 * nu * t)
    diff = _diffusion_scale(nu, sigma2, t)
    return decay * (R @ Sigma0 @ R.T) + diff * jnp.eye(d, dtype=Sigma0.dtype)


def snapshot_covariances(
    Sigma0: jax.Array,
    times: jax.Array,
    nu: float | jax.Array,
    sigma2: float | jax.Array,
    Omega: jax.Array,
) -> jax.Array:
    """Stack of covariances at each snapshot time, shape (K, d, d)."""
    def one(t):
        return ou_covariance(Sigma0, t, nu, sigma2, Omega)

    return jax.vmap(one)(times)


def lyapunov_rhs(H: jax.Array, Sigma: jax.Array, sigma2: float | jax.Array) -> jax.Array:
    """d Sigma / dt = H Sigma + Sigma H^T + 2 sigma^2 I."""
    d = Sigma.shape[0]
    sigma2 = jnp.asarray(sigma2, dtype=Sigma.dtype)
    return H @ Sigma + Sigma @ H.T + 2.0 * sigma2 * jnp.eye(d, dtype=Sigma.dtype)


def sample_centered_gaussian(key: jax.Array, n: int, Sigma: jax.Array) -> jax.Array:
    """n i.i.d. rows ~ N(0, Sigma)."""
    d = Sigma.shape[0]
    L = jnp.linalg.cholesky(Sigma)
    z = jax.random.normal(key, (n, d), dtype=Sigma.dtype)
    return z @ L.T


def sample_independent_slices(
    key: jax.Array,
    n: int,
    Sigmas: jax.Array,
) -> jax.Array:
    """Independent Gaussian slices, shape (K, n, d). Prop. 16 observation model."""
    K = Sigmas.shape[0]
    keys = jax.random.split(key, K)

    def one(k, Sigma):
        return sample_centered_gaussian(k, n, Sigma)

    return jax.vmap(one)(keys, Sigmas)
