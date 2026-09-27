"""Closed-form ZF (Prop 1) and collapsed precoder (Lemma 2)."""

from __future__ import annotations

import jax.numpy as jnp


def zf_precoder(A):
    """Eq (14): W_ZF = A^H (A A^H)^{-1}, so A W_ZF = I."""
    aah = A @ A.conj().T
    return A.conj().T @ jnp.linalg.inv(aah)


def zf_collapsed_X(A):
    """X^{(0)} = (A A^H)^{-1} so W = A^H X recovers ZF."""
    return jnp.linalg.inv(A @ A.conj().T)


def collapsed_precoder(A, X):
    """Lemma 2: W = A^H X."""
    return A.conj().T @ X


def frobenius_power(W):
    """||W||_F^2 as a real scalar."""
    return jnp.real(jnp.sum(jnp.abs(W) ** 2))


def power_normalize(W, Pmax):
    """W <- W * sqrt(Pmax / ||W||_F^2), differentiable for Pmax > 0."""
    power = frobenius_power(W)
    scale = jnp.sqrt(Pmax / jnp.maximum(power, jnp.asarray(1e-12, dtype=power.dtype)))
    return W * scale
