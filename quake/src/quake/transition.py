"""Shift-register transition and Lagrange-in-λ coefficients (paper eqs 8, 11, 81)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def e1(window: int, dtype=jnp.float32) -> Array:
    """First standard basis vector; paper `G = e_1`."""
    g = jnp.zeros((window,), dtype=dtype)
    return g.at[0].set(jnp.asarray(1.0, dtype=dtype))


def observation_row(gamma, window: int) -> Array:
    """`H_i = γ_i e_1^⊤` as a row vector (or batch of rows)."""
    gamma = jnp.asarray(gamma)
    return gamma[..., None] * e1(window, dtype=gamma.dtype)


def shift_register(coeffs, bias=0.0, window: int | None = None) -> tuple[Array, Array, Array]:
    """Build `A`, `b`, `G` for a length-`window` output state.

    State is `s = (f_i, f_{i-1}, ..., f_{i-H+1})`. The first row of `A` is the
    linear predictor for the new output; remaining rows shift history forward
    (paper eq 11). `bias` is the affine intercept of that predictor (Lagrange
    `x0` conversion), not the process-residual mean `μ`.
    """
    coeffs = jnp.asarray(coeffs)
    if window is None:
        window = int(coeffs.shape[-1])
    if coeffs.shape[-1] != window:
        raise ValueError(f"coeffs last dim {coeffs.shape[-1]} != window {window}")
    dtype = coeffs.dtype
    A = jnp.zeros(coeffs.shape[:-1] + (window, window), dtype=dtype)
    A = A.at[..., 0, :].set(coeffs)
    if window > 1:
        eye_shift = jnp.eye(window - 1, window, dtype=dtype)
        A = A.at[..., 1:, :].set(eye_shift)
    b = jnp.zeros(coeffs.shape[:-1] + (window,), dtype=dtype)
    b = b.at[..., 0].set(jnp.asarray(bias, dtype=dtype))
    G = e1(window, dtype=dtype)
    return A, b, G


def lagrange_coefficients(lambdas) -> Array:
    """Lagrange basis at `λ_i` using nodes `λ_{i-1}, ..., λ_{i-H}`.

    `lambdas[0]` is the query log-SNR `λ_i`; `lambdas[1:]` are the history
    nodes. Returns coefficients `a_k` so
    `f(λ_i) ≈ sum_k a_k f(λ_{i-k})` (degree `H-1` interpolant).
    """
    lambdas = jnp.asarray(lambdas)
    query = lambdas[..., 0]
    nodes = lambdas[..., 1:]
    h = nodes.shape[-1]
    # ℓ_k(λ) = Π_{j≠k} (λ − x_j) / (x_k − x_j)
    delta = nodes[..., :, None] - nodes[..., None, :]
    eye = jnp.eye(h, dtype=lambdas.dtype)
    denom = delta + eye
    num = query[..., None] - nodes
    num_mat = jnp.broadcast_to(num[..., None, :], denom.shape)
    num_mat = jnp.where(eye.astype(bool), jnp.asarray(1.0, dtype=lambdas.dtype), num_mat)
    return jnp.prod(num_mat / denom, axis=-1)


def lagrange_transition(lambdas, alpha=None, beta=None) -> tuple[Array, Array, Array]:
    """Shift-register `A, b, G` from Lagrange-in-λ, optional affine `x0` maps.

    If `alpha`/`beta` are given they are `(H+1,)` samples of the affine map
    `x0 = α f + β` at `(λ_i, λ_{i-1}, ..., λ_{i-H})`. The composed predictor
    stays affine in previous model outputs (Appendix D) and is absorbed into
    the first row of `A` and the first entry of `b`.
    """
    lambdas = jnp.asarray(lambdas)
    a = lagrange_coefficients(lambdas)
    window = int(a.shape[-1])
    if alpha is None:
        coeffs = a
        bias = jnp.asarray(0.0, dtype=a.dtype)
    else:
        alpha = jnp.asarray(alpha, dtype=a.dtype)
        beta = jnp.zeros_like(alpha) if beta is None else jnp.asarray(beta, dtype=a.dtype)
        alpha_i = alpha[..., 0]
        beta_i = beta[..., 0]
        coeffs = a * alpha[..., 1:] / alpha_i[..., None]
        bias = (jnp.sum(a * beta[..., 1:], axis=-1) - beta_i) / alpha_i
    return shift_register(coeffs, bias, window=window)
