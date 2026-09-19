"""Dirichlet log-density ratio (Eq 11/13) and Renyi divergence (Eq 12)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln


def log_beta(alpha: jax.Array) -> jax.Array:
    """log B(alpha) = sum_j gammaln(alpha_j) - gammaln(sum_j alpha_j)."""
    alpha = jnp.asarray(alpha)
    return jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))


def log_dirichlet_pdf(rho: jax.Array, alpha: jax.Array) -> jax.Array:
    """log Dir(rho | alpha) = -log B(alpha) + sum_j (alpha_j - 1) log(rho_j)."""
    rho = jnp.asarray(rho)
    alpha = jnp.asarray(alpha)
    return -log_beta(alpha) + jnp.sum((alpha - 1.0) * jnp.log(rho))


def log_density_ratio(rho: jax.Array, alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """Pointwise log-density ratio ln QS(rho)/P(rho) for QS=Dir(alpha), P=Dir(beta).

    Eq 11 / 13:
        ln QS(rho)/P(rho) = ln B(beta)/B(alpha) + sum_j (alpha_j - beta_j) ln(rho_j)
    """
    rho = jnp.asarray(rho)
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    return (log_beta(beta) - log_beta(alpha)) + jnp.sum((alpha - beta) * jnp.log(rho))


def dirichlet_renyi(alpha: jax.Array, beta: jax.Array, lam: float) -> jax.Array:
    """Dirichlet Renyi divergence D_lam(QS || P) of Eq 12.

        D_lam(QS || P) = ln B(beta)/B(alpha)
                         + 1/(lam-1) ln B(lam alpha + (1-lam) beta)/B(alpha)

    Finite iff alpha_j > ((lam-1)/lam) beta_j for all j (equivalently the mixed
    concentration lam alpha + (1-lam) beta is componentwise positive).
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    mix = lam * alpha + (1.0 - lam) * beta
    mix_safe = jnp.maximum(mix, jnp.array(1.0e-12, dtype=alpha.dtype))
    value = (log_beta(beta) - log_beta(alpha)) + (
        (log_beta(mix_safe) - log_beta(alpha)) / (lam - 1.0)
    )
    return jnp.where(jnp.all(mix > 0.0), value, jnp.array(jnp.inf, dtype=alpha.dtype))


def dirichlet_kl(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """KL(Dir(alpha) || Dir(beta)), the lam -> 1+ limit of Eq 12."""
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    a0 = jnp.sum(alpha)
    return (log_beta(beta) - log_beta(alpha)) + jnp.sum(
        (alpha - beta) * (digamma(alpha) - digamma(a0))
    )


def sample_dirichlet(key: jax.Array, alpha: jax.Array) -> jax.Array:
    """Draw rho ~ Dir(alpha) via jax.random.dirichlet (Algorithm 1)."""
    return jax.random.dirichlet(key, jnp.asarray(alpha))
