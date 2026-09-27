"""STAR-RIS coefficients, amplitude split, and composite channel."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def amplitude_split(phi):
    """Lemma 1: beta_t = cos phi, beta_r = sin phi, phi in [0, pi/2]."""
    return jnp.cos(phi), jnp.sin(phi)


def star_coefficients(phi, theta_t, theta_r):
    """Eq (1): vt = beta_t odot exp(j theta_t), vr = beta_r odot exp(j theta_r)."""
    beta_t, beta_r = amplitude_split(phi)
    vt = beta_t * jnp.exp(1j * theta_t)
    vr = beta_r * jnp.exp(1j * theta_r)
    return vt, vr


def unconstrained_phi(phi_tilde):
    """Prop 2: phi = (pi/2) sigmoid(phi_tilde) in (0, pi/2)."""
    return (jnp.pi / 2.0) * jax.nn.sigmoid(phi_tilde)


def phi_to_tilde(phi, eps=1e-4):
    """Inverse of Prop 2, with clipping to keep logit finite."""
    u = jnp.clip(phi / (jnp.pi / 2.0), eps, 1.0 - eps)
    return jnp.log(u) - jnp.log1p(-u)


def composite_channel(Ht, Hr, G, vt, vr):
    """Eq (6): A = [Ht diag(vt) G; Hr diag(vr) G] in C^{K x Nt}."""
    At = (Ht * vt) @ G
    Ar = (Hr * vr) @ G
    return jnp.concatenate([At, Ar], axis=0)
