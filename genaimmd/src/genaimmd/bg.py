"""Conditioned Boltzmann generator: loss (7), weights (8), RESS (9)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from genaimmd.committor import pB
from genaimmd.flow import forward, inverse
from genaimmd.potential import BETA, KBIAS, biased_energy, energy


def integer_bias_centers(q_vals):
    """Integer bias center closest to predicted committor in log-space.

    q is the log-committor. Default CV is ξ=pB, so the center is
    sigmoid(round(q)).
    """
    return jax.nn.sigmoid(jnp.round(q_vals))


def target_energy(x, xi_hat, committor_params, kbias: float = KBIAS):
    """Biased energy (10) with ξ = pB."""
    xi = pB(committor_params, x)
    return biased_energy(x, xi, xi_hat, kbias=kbias)


def log_prior(z):
    d = z.shape[-1]
    return -0.5 * jnp.sum(jnp.square(z), axis=-1) - 0.5 * d * jnp.log(2.0 * jnp.pi)


def bg_loss(
    flow_params,
    z,
    x_data,
    c_z,
    c_data,
    committor_params,
    lam_kl: float = 1.0,
    lam_ml: float = 1.0,
    beta: float = BETA,
    kbias: float = KBIAS,
):
    """L_BG = λ_KL E_z[βU(Fzx)-log|det Jzx|] + λ_ML E_x[½||Fxz||² - log|det Jxz|].

    U is the conditioned energy U'_ξ̂ when a committor is supplied.
    """
    x_gen, logdet_zx = forward(flow_params, z, c_z)
    u = target_energy(x_gen, c_z, committor_params, kbias=kbias)
    kl = jnp.mean(beta * u - logdet_zx)
    z_enc, logdet_xz = inverse(flow_params, x_data, c_data)
    ml = jnp.mean(0.5 * jnp.sum(jnp.square(z_enc), axis=-1) - logdet_xz)
    return lam_kl * kl + lam_ml * ml


def log_importance_weights(flow_params, z, c, beta: float = BETA):
    """log ω(Fzx(z)) from eq. (8), up to a constant. Uses unbiased U."""
    x, logdet = forward(flow_params, z, c)
    u = energy(x)
    return -beta * u - log_prior(z) + logdet


def importance_weights(flow_params, z, c, beta: float = BETA):
    logw = log_importance_weights(flow_params, z, c, beta=beta)
    logw = logw - jnp.max(logw)
    # clip so untrained maps with huge U still give strictly positive weights
    return jnp.exp(jnp.maximum(logw, -80.0))


def ress(weights):
    """Relative effective sample size, eq. (9): (1/N)(∑ω)² / ∑ω²."""
    w = jnp.asarray(weights)
    n = w.shape[0]
    return (jnp.sum(w) ** 2) / (n * jnp.sum(jnp.square(w)))
