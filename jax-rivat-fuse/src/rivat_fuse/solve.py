"""Variational latent solve H h = b (RiVaT-Fuse eqs 4-6).

H = P_I + P_M + λ I
b = P_I z_I + P_M z_M + T(z_I ⊗ z_M)
h* = H^{-1} b  (via a linear solve, not an explicit inverse)
"""

from __future__ import annotations

import jax.numpy as jnp


def assemble_H(P_I, P_M, lam):
    dim = P_I.shape[-1]
    return P_I + P_M + lam * jnp.eye(dim, dtype=P_I.dtype)


def assemble_b(P_I, P_M, z_I, z_M, t_vec):
    return P_I @ z_I + P_M @ z_M + t_vec


def solve_latent(H, b):
    """Solve H h = b. H is SPD when P_I, P_M ⪰ 0 and λ > 0 (or ε > 0)."""
    return jnp.linalg.solve(H, b)


def quadratic_objective(h, H, b):
    """1/2 h^T H h - b^T h, the strongly convex objective whose minimizer is h*."""
    return 0.5 * (h @ (H @ h)) - b @ h
