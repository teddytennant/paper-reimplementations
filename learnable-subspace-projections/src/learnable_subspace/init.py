"""Whitened SVD initialization for input-side and output-side projectors."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from learnable_subspace.projector import orthogonal_factor


class WhitenedInit(NamedTuple):
    """Input-side whitened init.

    V is the unconstrained basis of dropped directions, shape (din, din - r).
    P_init projects onto span(C_r). sigma are singular values of W @ S.
    """

    V: jax.Array
    P_init: jax.Array
    C_r: jax.Array
    sigma: jax.Array
    S: jax.Array


def empirical_gram(X):
    """G = (1/n) X.T @ X for calibration X of shape (n, din)."""
    n = X.shape[0]
    return (X.T @ X) / n


def cholesky_covariance(X, ridge_eps=1e-6):
    """Lower-triangular S with G = S @ S.T, G = (1/n) X.T @ X.

    JAX Cholesky returns NaNs instead of raising when G is not positive
    definite. On failure, ridge H = X.T @ X by eta = -min_eig(H) + ridge_eps
    and factor the ridged Gram H_ridge / n.
    """
    n, d = X.shape
    H = X.T @ X
    G = H / n
    S = jnp.linalg.cholesky(G)
    if bool(jnp.all(jnp.isfinite(S))):
        return S
    eta = -jnp.min(jnp.linalg.eigvalsh(H)) + ridge_eps
    H_r = H + eta * jnp.eye(d, dtype=H.dtype)
    return jnp.linalg.cholesky(H_r / n)


def tail_energy(sigma, r):
    """sum_{i > r} sigma_i^2, with sigma descending and 0-based slice sigma[r:]."""
    return jnp.sum(jnp.asarray(sigma)[r:] ** 2)


def reconstruction_energy(W, Z, S):
    """E(Z) = ||(W - Z) @ S||_F^2."""
    residual = (W - Z) @ S
    return jnp.sum(residual ** 2)


def whitened_input_init(W, X, r):
    """Drop the trailing right singular directions of the whitened weight.

    SVD of W @ S = U_w @ diag(sigma) @ V_w.T.
    basis = S^{-T} @ V_w[:, r:], then V_init = qf(basis).
    C_r = S @ V_w[:, :r], P_init = C_r @ pinv(C_r).
    r is the number of kept input directions. k = din - r are dropped.
    """
    dout, din = W.shape
    del dout
    if r < 0 or r > din:
        raise ValueError(f"r must lie in [0, din={din}], got {r}")
    S = cholesky_covariance(X)
    _U_w, sigma, Vh = jnp.linalg.svd(W @ S, full_matrices=True)
    V_w = Vh.T
    dropped = V_w[:, r:]
    if dropped.shape[1] == 0:
        V = jnp.zeros((din, 0), dtype=W.dtype)
    else:
        basis = jnp.linalg.solve(S.T, dropped)
        V = orthogonal_factor(basis)
    if r == 0:
        C_r = jnp.zeros((din, 0), dtype=W.dtype)
        P_init = jnp.zeros((din, din), dtype=W.dtype)
    else:
        C_r = S @ V_w[:, :r]
        P_init = C_r @ jnp.linalg.pinv(C_r)
    return WhitenedInit(V=V, P_init=P_init, C_r=C_r, sigma=sigma, S=S)


def output_side_init(W, k):
    """Removed output directions: trailing left singular vectors of W.

    No input covariance factor is applied. V has shape (dout, k).
    """
    dout, _din = W.shape
    if k < 0 or k > dout:
        raise ValueError(f"k must lie in [0, dout={dout}], got {k}")
    if k == 0:
        return jnp.zeros((dout, 0), dtype=W.dtype)
    U_w, _sigma, _Vh = jnp.linalg.svd(W, full_matrices=True)
    return U_w[:, dout - k :]
