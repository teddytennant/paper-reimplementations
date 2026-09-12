"""Coefficient paths: cosine VP and Cond-OT."""

from __future__ import annotations

import jax.numpy as jnp

SOFF = 0.008


def phi_cosine(t, soff: float = SOFF):
    """Cosine phase φ(t) = (π/2) (t + soff) / (1 + soff)."""
    t = jnp.asarray(t)
    return (jnp.pi / 2.0) * (t + soff) / (1.0 + soff)


def alphabar(t, soff: float = SOFF):
    """ᾱ(t) = cos(φ(t))² / cos(φ(0))²."""
    t = jnp.asarray(t)
    phi_t = phi_cosine(t, soff=soff)
    phi_0 = phi_cosine(jnp.zeros((), dtype=t.dtype), soff=soff)
    c0 = jnp.cos(phi_0)
    return jnp.square(jnp.cos(phi_t)) / jnp.square(c0)


def alpha_sigma(t, soff: float = SOFF):
    """α(t) = √ᾱ, σ(t) = √(1 − ᾱ)."""
    ab = alphabar(t, soff=soff)
    ab = jnp.clip(ab, 0.0, 1.0)
    alpha = jnp.sqrt(ab)
    sigma = jnp.sqrt(1.0 - ab)
    return alpha, sigma


def cosine_vp_arc_length(t_edges, soff: float = SOFF):
    """True VP arc length (D.3): L_k = arccos(√ᾱ(τ_{k+1})) − arccos(√ᾱ(τ_k)).

    τ increases data→noise so ᾱ decreases and L_k is positive.
    """
    t_edges = jnp.asarray(t_edges)
    ab = jnp.clip(alphabar(t_edges, soff=soff), 0.0, 1.0)
    theta = jnp.arccos(jnp.sqrt(ab))
    return theta[1:] - theta[:-1]


def cosine_phase_increment(t_edges, soff: float = SOFF):
    """Practical DDPM increment (D.4): Δφ_k = φ(τ_{k+1}) − φ(τ_k)."""
    t_edges = jnp.asarray(t_edges)
    phi = phi_cosine(t_edges, soff=soff)
    return phi[1:] - phi[:-1]


def cond_ot_gamma(tau):
    """γ(τ) = (1 − τ, τ)."""
    tau = jnp.asarray(tau)
    return 1.0 - tau, tau


def cond_ot_arc_length(tau_edges):
    """L_k = √2 (τ_{k+1} − τ_k)."""
    tau_edges = jnp.asarray(tau_edges)
    return jnp.sqrt(2.0) * (tau_edges[1:] - tau_edges[:-1])
