"""Associative scan of affine maps (Eq. 4, Prop. B.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def affine_compose(left, right):
    """Composition (F', g') ∘ (F, g) := (F' F, F' g + g') with left=(F,g), right=(F',g').

    `associative_scan` feeds earlier-in-time as `left` and later-in-time as `right`,
    so this applies the earlier map first.
    """
    F_l, g_l = left
    F_r, g_r = right
    F = F_r @ F_l
    g = jnp.einsum("...ij,...j->...i", F_r, g_l) + g_r
    return F, g


def apply_affine(F, z, g):
    """z ↦ F z + g."""
    return jnp.einsum("...ij,...j->...i", F, z) + g


def sequential_simulate(F, g, z0):
    """Sequential affine recurrence Z_{k+1} = F_k Z_k + g_k (Eq. 3).

    Args:
        F: (B, N, d, d)
        g: (B, N, d)
        z0: (B, d)
    Returns:
        path Z of shape (B, N+1, d)
    """
    F_t = jnp.swapaxes(F, 0, 1)
    g_t = jnp.swapaxes(g, 0, 1)

    def step(z, fg):
        Fk, gk = fg
        z = apply_affine(Fk, z, gk)
        return z, z

    _, zs = jax.lax.scan(step, z0, (F_t, g_t))
    return jnp.concatenate([z0[:, None, :], jnp.swapaxes(zs, 0, 1)], axis=1)


def parallel_simulate(F, g, z0):
    """Parallel associative scan of the same recurrence (Eq. 4, Prop. B.1).

    Inclusive scan of the N transitions yields maps Z_0 → Z_1, …, Z_N.
    """
    F_pref, g_pref = jax.lax.associative_scan(affine_compose, (F, g), axis=1)
    z_rest = apply_affine(F_pref, z0[:, None, :], g_pref)
    return jnp.concatenate([z0[:, None, :], z_rest], axis=1)
