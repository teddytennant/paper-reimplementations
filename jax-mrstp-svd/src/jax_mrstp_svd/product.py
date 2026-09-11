"""Tensor / matrix semi-tensor products and the L-induced t-product.

Implements Definition 2.2–2.3, Lemma 2.2, Definition 3.1, Lemma 3.1,
Remark 3.1, and Theorem 3.1 of arXiv:2609.11168.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from .transforms import Transform, get_transform


def frobenius_norm(A: Array) -> Array:
    return jnp.linalg.norm(A.reshape(-1), ord=2)


def matrix_stp(A: Array, B: Array) -> Array:
    """Left matrix STP (Definition 2.2 / Lemma 2.2).

    For A ∈ R^{m×n}, B ∈ R^{p×q} and t = lcm(n, p):
        A ⋉ B = (A ⊗ I_{t/n}) (B ⊗ I_{t/p}).
    When n = p this reduces to AB (Remark 2.1).
    """
    _m, n = A.shape
    p, _q = B.shape
    t = math.lcm(int(n), int(p))
    A_exp = jnp.kron(A, jnp.eye(t // int(n), dtype=A.dtype))
    B_exp = jnp.kron(B, jnp.eye(t // int(p), dtype=B.dtype))
    return A_exp @ B_exp


def tensor_kronecker(A: Array, B: Array) -> Array:
    """Third-order tensor Kronecker product (Definition S1.2).

    C[i1 j1, i2 j2, i3 j3] = A_{i1 i2 i3} B_{j1 j2 j3}.
    """
    n1, n2, n3 = A.shape
    m1, m2, m3 = B.shape
    out = A[:, None, :, None, :, None] * B[None, :, None, :, None, :]
    return out.reshape(n1 * m1, n2 * m2, n3 * m3)


def t_product(A: Array, B: Array, transform: str | Transform = "fft") -> Array:
    """t-product A *_L B (Definition 2.3): slice-wise matmul in the L-domain."""
    L = get_transform(transform)
    A_bar = L.forward(A)
    B_bar = L.forward(B)
    C_bar = jnp.einsum("ijk,jlk->ilk", A_bar, B_bar)
    return L.inverse(C_bar)


def tensor_stp(A: Array, B: Array, transform: str | Transform = "fft") -> Array:
    """Tensor STP A ⋉_L B (Definition 3.1, computed via Lemma 3.1).

    Slice-wise matrix STP in the transform domain, then L^{-1}.
    If n2 = m1 this equals the t-product (Remark 3.1).
    """
    L = get_transform(transform)
    A_bar = L.forward(A)
    B_bar = L.forward(B)
    n3 = A_bar.shape[2]
    slices = [matrix_stp(A_bar[:, :, j], B_bar[:, :, j]) for j in range(n3)]
    C_bar = jnp.stack(slices, axis=2)
    return L.inverse(C_bar)


def tensor_stp_by_definition(
    A: Array, B: Array, transform: str | Transform = "fft"
) -> Array:
    """Definition 3.1 literally: (A ⊗ I_{t/n2}) *_L (B ⊗ I_{t/m1})."""
    _n1, n2, _n3 = A.shape
    m1, _m2, _ = B.shape
    t = math.lcm(int(n2), int(m1))
    I_left = jnp.eye(t // int(n2), dtype=A.dtype)[:, :, None]
    I_right = jnp.eye(t // int(m1), dtype=B.dtype)[:, :, None]
    return t_product(tensor_kronecker(A, I_left), tensor_kronecker(B, I_right), transform)


def tensor_transpose(A: Array, transform: str | Transform = "fft") -> Array:
    """Tensor transpose under L (Definition 2.4, conjugate on complex slices)."""
    L = get_transform(transform)
    A_bar = L.forward(A)
    A_bar_t = jnp.conj(jnp.transpose(A_bar, (1, 0, 2)))
    return L.inverse(A_bar_t)


def identity_tensor(
    n: int,
    n3: int,
    transform: str | Transform = "fft",
    dtype=jnp.float64,
) -> Array:
    """Identity tensor under L (Definition 2.5): L(I)^{(i)} = I_n."""
    L = get_transform(transform)
    I_bar = jnp.broadcast_to(jnp.eye(n, dtype=dtype)[:, :, None], (n, n, n3))
    return L.inverse(I_bar)
