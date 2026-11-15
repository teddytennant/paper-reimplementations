"""Algebraic identities from arXiv:2609.11168 (Sections 2–3)."""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import random

from jax_mrstp_svd.product import (
    identity_tensor,
    matrix_stp,
    t_product,
    tensor_kronecker,
    tensor_stp,
    tensor_stp_by_definition,
    tensor_transpose,
)
from jax_mrstp_svd.rearrange import kronecker_terms, rearrange, unrearrange


def _rand(key, shape, dtype=jnp.float64):
    return random.normal(key, shape, dtype=dtype)


def test_matrix_stp_reduces_to_matmul_when_inner_dims_match():
    """Remark 2.1: n = p ⇒ A ⋉ B = AB."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    A = _rand(k1, (4, 3))
    B = _rand(k2, (3, 5))
    assert jnp.allclose(matrix_stp(A, B), A @ B)


def test_matrix_stp_associative():
    """Lemma 2.1."""
    key = random.PRNGKey(1)
    keys = random.split(key, 3)
    A = _rand(keys[0], (2, 3))
    B = _rand(keys[1], (6, 2))
    C = _rand(keys[2], (4, 5))
    left = matrix_stp(matrix_stp(A, B), C)
    right = matrix_stp(A, matrix_stp(B, C))
    assert left.shape == right.shape
    assert jnp.allclose(left, right)


def test_matrix_stp_kronecker_formula():
    """Lemma 2.2."""
    A = _rand(random.PRNGKey(2), (3, 2))
    B = _rand(random.PRNGKey(3), (4, 5))
    n, p = 2, 4
    t = math.lcm(n, p)
    formula = jnp.kron(A, jnp.eye(t // n)) @ jnp.kron(B, jnp.eye(t // p))
    assert jnp.allclose(matrix_stp(A, B), formula)


def test_tensor_stp_reduces_to_tproduct():
    """Remark 3.1: n2 = m1 ⇒ A ⋉_L B = A *_L B."""
    key = random.PRNGKey(4)
    k1, k2 = random.split(key)
    A = _rand(k1, (3, 4, 5))
    B = _rand(k2, (4, 2, 5))
    for transform in ("identity", "fft", "dft"):
        stp = tensor_stp(A, B, transform)
        tp = t_product(A, B, transform)
        assert stp.shape == tp.shape == (3, 2, 5)
        assert jnp.allclose(stp, tp, rtol=1e-10, atol=1e-10), transform


def test_definition_3_1_matches_lemma_3_1():
    """Definition 3.1 vs Lemma 3.1 (slice-wise matrix STP)."""
    A = _rand(random.PRNGKey(5), (2, 3, 4))
    B = _rand(random.PRNGKey(6), (6, 2, 4))
    for transform in ("identity", "fft"):
        via_slices = tensor_stp(A, B, transform)
        via_def = tensor_stp_by_definition(A, B, transform)
        assert via_slices.shape == via_def.shape == (4, 2, 4)
        assert jnp.allclose(via_slices, via_def, rtol=1e-10, atol=1e-10), transform


def test_tensor_stp_associative():
    """Theorem 3.1."""
    keys = random.split(random.PRNGKey(7), 3)
    A = _rand(keys[0], (2, 3, 4))
    B = _rand(keys[1], (6, 2, 4))
    C = _rand(keys[2], (4, 5, 4))
    for transform in ("identity", "fft"):
        left = tensor_stp(tensor_stp(A, B, transform), C, transform)
        right = tensor_stp(A, tensor_stp(B, C, transform), transform)
        assert left.shape == right.shape == (8, 5, 4)
        assert jnp.allclose(left, right, rtol=1e-9, atol=1e-9), transform


def test_tproduct_identity():
    """Definition 2.5: A *_L I = A."""
    A = _rand(random.PRNGKey(8), (3, 3, 4))
    for transform in ("identity", "fft"):
        I = identity_tensor(3, 4, transform, dtype=A.dtype)
        AI = t_product(A, I, transform)
        IA = t_product(I, A, transform)
        assert jnp.allclose(AI, A, rtol=1e-10, atol=1e-10), transform
        assert jnp.allclose(IA, A, rtol=1e-10, atol=1e-10), transform


def test_rearrange_roundtrip_and_kronecker_rank1():
    """Eq. (2): R(B ⊗ C) = vec(B) vec(C)^T (column-major vec)."""
    m1, m2, n1, n2 = 2, 3, 2, 4
    B = _rand(random.PRNGKey(9), (m1, n1))
    C = _rand(random.PRNGKey(10), (m2, n2))
    A = jnp.kron(B, C)
    R = rearrange(A, m1, m2, n1, n2)
    vecB = B.T.reshape(-1)
    vecC = C.T.reshape(-1)
    assert jnp.allclose(R, jnp.outer(vecB, vecC))
    assert jnp.allclose(unrearrange(R, m1, m2, n1, n2), A)


def test_kronecker_terms_recover_sum():
    """Lemma 4.2: rank-k SVD of R recovers Σ B_i ⊗ C_i."""
    m1, m2, n1, n2, k = 3, 2, 3, 2, 2
    keys = random.split(random.PRNGKey(11), 2 * k)
    A = jnp.zeros((m1 * m2, n1 * n2))
    for i in range(k):
        B = _rand(keys[2 * i], (m1, n1))
        C = _rand(keys[2 * i + 1], (m2, n2))
        A = A + jnp.kron(B, C)
    rec = sum(jnp.kron(B, C) for B, C in kronecker_terms(A, m1, m2, n1, n2, k))
    assert jnp.allclose(rec, A, rtol=1e-10, atol=1e-10)


def test_fft_matches_explicit_dft():
    A = _rand(random.PRNGKey(12), (2, 3, 4))
    B = _rand(random.PRNGKey(13), (6, 2, 4))
    assert jnp.allclose(
        tensor_stp(A, B, "fft"), tensor_stp(A, B, "dft"), rtol=1e-10, atol=1e-10
    )


def test_tensor_transpose_involution():
    A = _rand(random.PRNGKey(14), (2, 5, 3))
    for transform in ("identity", "fft"):
        At = tensor_transpose(A, transform)
        assert At.shape == (5, 2, 3)
        assert jnp.allclose(tensor_transpose(At, transform), A, rtol=1e-10, atol=1e-10)
