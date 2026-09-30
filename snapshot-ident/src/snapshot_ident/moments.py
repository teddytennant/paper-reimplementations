"""Test functions and the weak-moment design of eq. 6."""
from __future__ import annotations

from . import _host

_host.preload()

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def second_moments_from_samples(xs: jax.Array) -> jax.Array:
    """Empirical E[x x^T] from a slice of shape (n, d)."""
    n = xs.shape[0]
    return (xs.T @ xs) / jnp.asarray(n, dtype=xs.dtype)


def stacked_second_moments(slices: jax.Array) -> jax.Array:
    """slices (K, n, d) -> second-moment stack (K, d, d)."""
    return jax.vmap(second_moments_from_samples)(slices)


def e_psi_sqnorm(M: jax.Array, mu: jax.Array | None = None) -> jax.Array:
    """E[||x||^2] = tr(E[xx^T]). If mu is given, M is the central covariance."""
    val = jnp.trace(M)
    if mu is not None:
        val = val + mu @ mu
    return val


def e_psi_coord(mu: jax.Array, i: int = 0) -> jax.Array:
    """E[x_i]."""
    return mu[i]


def e_generator_sqnorm(
    H: jax.Array,
    M: jax.Array,
    sigma2: float | jax.Array,
    mu: jax.Array | None = None,
) -> jax.Array:
    """E[F · ∇ψ + σ^2 Δψ] for ψ = ||x||^2 and F(x) = H x.

    ∇ψ = 2x, Δψ = 2d, so the expectation is 2 E[x^T H x] + 2 d σ^2.
    """
    d = H.shape[0]
    sigma2 = jnp.asarray(sigma2, dtype=H.dtype)
    if mu is None:
        exhx = jnp.trace(H @ M)
    else:
        exhx = mu @ (H @ mu) + jnp.trace(H @ M)
    return 2.0 * exhx + 2.0 * d * sigma2


def e_generator_coord(
    H: jax.Array,
    mu: jax.Array,
    i: int = 0,
) -> jax.Array:
    """E[F · ∇ψ + σ^2 Δψ] for ψ = x_i. Laplacian vanishes, E[F_i] = (H mu)_i."""
    return (H @ mu)[i]


def ito_sqnorm_pair(
    H: jax.Array,
    Sigma: jax.Array,
    sigma2: float | jax.Array,
    mu: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """(d/dt E[||x||^2], E[F·∇ψ + σ^2 Δψ]) for a Gaussian with cov Sigma."""
    dSigma = H @ Sigma + Sigma @ H.T + 2.0 * jnp.asarray(sigma2, dtype=Sigma.dtype) * jnp.eye(
        Sigma.shape[0], dtype=Sigma.dtype
    )
    d_mu_term = 0.0
    if mu is not None:
        d_mu_term = 2.0 * mu @ (H @ mu)
    ddt = jnp.trace(dSigma) + d_mu_term
    gen = e_generator_sqnorm(H, Sigma, sigma2, mu=mu)
    return ddt, gen


def ito_coord_pair(
    H: jax.Array,
    mu: jax.Array,
    i: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """(d/dt E[x_i], E[F·∇ψ + σ^2 Δψ]) for ψ = x_i."""
    ddt = (H @ mu)[i]
    gen = e_generator_coord(H, mu, i=i)
    return ddt, gen


def trapezoidal_residual(
    psi: jax.Array,
    generator: jax.Array,
    dts: jax.Array,
) -> jax.Array:
    """Eq. 6 residual on each interval: ΔE[ψ] - (Δ/2)(E_k + E_{k+1})[generator]."""
    return psi[1:] - psi[:-1] - 0.5 * dts * (generator[:-1] + generator[1:])


def _interval_design(M0: jax.Array, M1: jax.Array, dt: jax.Array, sigma2: jax.Array):
    """One-interval design for quadratic probes ψ_ab = x_a x_b and linear maps.

    bb_ab = E_{k+1}[ψ_ab] - E_k[ψ_ab] - (Δ/2)(E_k+E_{k+1})[σ^2 Δψ_ab]
    X_{ab, uv} = (Δ/2)(E_k+E_{k+1})[ ϕ^{uv} · ∇ψ_ab ]
    with ϕ^{uv}(x) = x_v e_u, i.e. the (u,v) matrix unit.
    """
    d = M0.shape[0]
    eye = jnp.eye(d, dtype=M0.dtype)
    lap = 2.0 * sigma2 * eye
    bb = (M1 - M0 - dt * lap).reshape(-1)
    Mbar = 0.5 * (M0 + M1)
    term1 = jnp.einsum("ua,vb->abuv", eye, Mbar)
    term2 = jnp.einsum("ub,va->abuv", eye, Mbar)
    X = (dt * (term1 + term2)).reshape(d * d, d * d)
    return X, bb


def weak_design_from_second_moments(
    moments: jax.Array,
    dts: jax.Array,
    sigma2: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Stack (X, bb) of eq. 6 over snapshot intervals for linear F(x) = H x.

    moments: (K, d, d) second-moment matrices E[x x^T] at each slice.
    dts: (K-1,) gaps Δk.
    Returns X of shape ((K-1) d^2, d^2) and bb of shape ((K-1) d^2,).
    Column p of X is the (u,v) feature with u, v in row-major order.
    """
    sigma2 = jnp.asarray(sigma2, dtype=moments.dtype)
    dts = jnp.asarray(dts, dtype=moments.dtype)
    M0 = moments[:-1]
    M1 = moments[1:]
    X, bb = jax.vmap(lambda a, b, dt: _interval_design(a, b, dt, sigma2))(M0, M1, dts)
    d = moments.shape[-1]
    return X.reshape(-1, d * d), bb.reshape(-1)


def weak_design_from_samples(
    slices: jax.Array,
    dts: jax.Array,
    sigma2: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Eq. 6 design from independent empirical slices of shape (K, n, d)."""
    moments = stacked_second_moments(slices)
    return weak_design_from_second_moments(moments, dts, sigma2)
