"""Van Loan–Pitsianis rearrangement and Kronecker terms (paper eq. (2), Lemmas 4.1–4.2)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def rearrange(A: Array, m1: int, m2: int, n1: int, n2: int) -> Array:
    """R(A) ∈ C^{m1 n1 × m2 n2} (eq. (2)).

    A is viewed as an m1×n1 block matrix with m2×n2 blocks.  The row of R
    indexed by block (i, j) (i = 0..m1-1, j = 0..n1-1) is vec(A_{i,j})^T
    with column-major vec, stacked as [A_0; A_1; ...; A_{n1-1}].
    """
    if A.shape != (m1 * m2, n1 * n2):
        raise ValueError(
            f"A has shape {tuple(A.shape)}, expected {(m1 * m2, n1 * n2)} "
            f"for blocks ({m1},{m2})×({n1},{n2})"
        )
    blocks = A.reshape(m1, m2, n1, n2)
    # R[j, i, q, p] = blocks[i, p, j, q]  →  (n1 m1, n2 m2)
    return jnp.transpose(blocks, (2, 0, 3, 1)).reshape(n1 * m1, n2 * m2)


def unrearrange(R: Array, m1: int, m2: int, n1: int, n2: int) -> Array:
    """Inverse of :func:`rearrange`."""
    tmp = R.reshape(n1, m1, n2, m2)
    return jnp.transpose(tmp, (1, 3, 0, 2)).reshape(m1 * m2, n1 * n2)


def _vec_to_mat(vec: Array, rows: int, cols: int) -> Array:
    """Column-major reshape of a vector into a matrix."""
    return vec.reshape((cols, rows)).T


def kronecker_terms(
    A: Array,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    k: int,
) -> list[tuple[Array, Array]]:
    """Leading k Kronecker terms of A (Lemmas 4.1–4.2).

    Returns ``[(B_i, C_i)]`` with B_i ∈ C^{m1×n1}, C_i ∈ C^{m2×n2} such that
    ``A ≈ Σ_i B_i ⊗ C_i`` and vec(B_i) = √σ_i U(:,i), vec(C_i) = √σ_i V(:,i)
    for the SVD of R(A).
    """
    R = rearrange(A, m1, m2, n1, n2)
    U, s, Vh = jnp.linalg.svd(R, full_matrices=False)
    k = min(int(k), int(s.shape[0]))
    terms: list[tuple[Array, Array]] = []
    for i in range(k):
        scale = jnp.sqrt(s[i])
        B = _vec_to_mat(scale * U[:, i], m1, n1)
        C = _vec_to_mat(scale * jnp.conj(Vh[i, :]), m2, n2)
        terms.append((B, C))
    return terms


def randomized_range_svd(
    R: Array,
    k: int,
    oversampling: int,
    n_power: int,
    G: Array,
) -> tuple[Array, Array, Array]:
    """Power-iterated randomized SVD of the rearranged matrix (Algorithm 5, lines 5–9).

    ``G`` is an (n_cols(R) × (k+s)) Gaussian sketch.  Returns thin factors
    ``(Uk, sk, Vhk)`` of rank ``k`` approximating ``R ≈ Uk diag(sk) Vhk``.
    """
    k = int(k)
    Y = R @ G
    if n_power > 0:
        Rt = jnp.conj(R.T)
        for _ in range(int(n_power)):
            Y = R @ (Rt @ Y)
    Q, _ = jnp.linalg.qr(Y, mode="reduced")
    B = jnp.conj(Q.T) @ R
    Uhat, s, Vh = jnp.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    return U[:, :k], s[:k], Vh[:k, :]


def kronecker_terms_from_svd(
    U: Array,
    s: Array,
    Vh: Array,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    k: int,
) -> list[tuple[Array, Array]]:
    """Build Kronecker factors from an SVD of R(A) (Algorithm 5, line 11)."""
    k = min(int(k), int(s.shape[0]))
    terms: list[tuple[Array, Array]] = []
    for i in range(k):
        scale = jnp.sqrt(s[i])
        B = _vec_to_mat(scale * U[:, i], m1, n1)
        C = _vec_to_mat(scale * jnp.conj(Vh[i, :]), m2, n2)
        terms.append((B, C))
    return terms
