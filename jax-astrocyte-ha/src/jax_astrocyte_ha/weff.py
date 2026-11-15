"""Effective N-RNN recurrent weights (Eq. 6)."""

from __future__ import annotations

import jax.numpy as jnp


def effective_weights(J_xx, H_px, pi):
    """W_eff(π) = J_xx ⊙ 11^T + H_px diag(π) H_px^T.

    Parameters
    ----------
    J_xx : (Nx, Nx)
        Baseline recurrent matrix.
    H_px : (Nx, M)
        Tall astrocytic tiling / projection, M << Nx.
    pi : (M,)
        Context vector, typically one-hot.

    Returns
    -------
    Weff : (Nx, Nx)
    """
    J_xx = jnp.asarray(J_xx)
    H_px = jnp.asarray(H_px)
    pi = jnp.asarray(pi).reshape(-1)
    ones = jnp.ones(J_xx.shape, dtype=J_xx.dtype)
    mod = H_px @ jnp.diag(pi) @ H_px.T
    return J_xx * ones + mod
