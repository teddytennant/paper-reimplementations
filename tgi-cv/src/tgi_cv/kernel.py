"""Gaussian incremental proposal and TGI random-walk Metropolis."""

from __future__ import annotations

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

import jax
import jax.numpy as jnp

from tgi_cv.hermite import mode_order


def beta_from_gamma(gamma):
    """beta = 1 - gamma, with gamma in (0, 1]."""
    return 1.0 - gamma


def eigenvalue(k, beta):
    """lambda_k = beta**|k|."""
    return jnp.asarray(beta) ** mode_order(k)


def mode_eigenvalues(modes, beta):
    beta = jnp.asarray(beta)
    modes = list(modes)
    if not modes:
        return jnp.zeros((0,), dtype=beta.dtype)
    orders = jnp.array([mode_order(k) for k in modes], dtype=beta.dtype)
    return beta ** orders


def asymptotic_factor(lam):
    """Empty-dictionary factor v_j = (1 + lambda_j) / (1 - lambda_j)."""
    lam = jnp.asarray(lam)
    return (1.0 + lam) / (1.0 - lam)


def gi_propose(z, xi, gamma):
    """GI proposal, z' = beta*z + sqrt(1-beta**2)*xi, beta = 1-gamma."""
    z = jnp.asarray(z)
    xi = jnp.asarray(xi)
    beta = beta_from_gamma(gamma)
    z_prop = beta * z + jnp.sqrt(1.0 - jnp.square(beta)) * xi
    return z_prop, beta


def acceptance_probability(log_w, log_w_prop):
    """min(1, w(z')/w(z)), returned as alpha and log alpha."""
    log_alpha = jnp.minimum(0.0, log_w_prop - log_w)
    return jnp.exp(log_alpha), log_alpha


def mh_accept(key, log_w, log_w_prop):
    """Draw the Metropolis accept bit. Uniform is on [0, 1)."""
    alpha, log_alpha = acceptance_probability(log_w, log_w_prop)
    u = jax.random.uniform(key, shape=jnp.shape(log_alpha))
    accept = jnp.log(u) < log_alpha
    return accept, alpha


def tgi_rwm_step(key, z, log_w, log_w_fn, gamma):
    """Propose with the GI kernel and accept with min(1, w(z')/w(z))."""
    key_xi, key_u = jax.random.split(key)
    xi = jax.random.normal(key_xi, jnp.shape(z))
    z_prop, beta = gi_propose(z, xi, gamma)
    log_w_prop = jnp.asarray(log_w_fn(z_prop), dtype=z.dtype)
    accept, alpha = mh_accept(key_u, log_w, log_w_prop)
    z_next = jnp.where(accept, z_prop, z)
    log_w_next = jnp.where(accept, log_w_prop, log_w)
    return z_next, log_w_next, z_prop, alpha, accept, beta
