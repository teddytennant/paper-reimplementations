"""MSTP-SVD (Algorithm 3 / Theorem 4.3) and MRSTP-SVD (Algorithm 5 / Theorem 5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random

from .product import frobenius_norm, tensor_stp, tensor_transpose
from .rearrange import (
    kronecker_terms,
    kronecker_terms_from_svd,
    randomized_range_svd,
    rearrange,
)
from .transforms import Transform, get_transform


@dataclass
class Factor:
    """One orthogonal STP term U ⋉_L S ⋉_L V^⊤ (Theorem 4.3)."""

    U: Array
    S: Array
    V: Array


def _svd_factors(B: Array, C: Array, rank: int | None) -> tuple[Array, Array, Array]:
    """Lines 5–6 of Algorithm 3 (or 12–13 of Algorithm 5): SVD of B, Σ = Σ_B ⊗ C.

    Full SVD (``rank is None``) returns square orthogonal U, V as in Algorithms 1–3.
    A positive ``rank`` keeps a thin truncated SVD of B (Algorithms 6–9).
    """
    if rank is None:
        U, s, Vh = jnp.linalg.svd(B, full_matrices=True)
        m, n = int(B.shape[0]), int(B.shape[1])
        Sigma_B = jnp.zeros((m, n), dtype=B.dtype)
        p = int(s.shape[0])
        Sigma_B = Sigma_B.at[:p, :p].set(jnp.diag(s.astype(B.dtype)))
        return U, jnp.kron(Sigma_B, C), jnp.conj(Vh.T)

    U, s, Vh = jnp.linalg.svd(B, full_matrices=False)
    r = min(int(rank), int(s.shape[0]))
    U, s, Vh = U[:, :r], s[:r], Vh[:r, :]
    Sigma_B = jnp.diag(s.astype(B.dtype))
    return U, jnp.kron(Sigma_B, C), jnp.conj(Vh.T)


def _stack_inverse(slices: list[Array], transform: Transform) -> Array:
    return transform.inverse(jnp.stack(slices, axis=2))


def mstp_svd(
    A: Array,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    k: int,
    transform: str | Transform = "fft",
    rank: int | None = None,
) -> list[Factor]:
    """Multi-term STP-SVD of a third-order tensor (Algorithm 3, Theorem 4.3).

    ``A`` has shape ``(m1 m2, n1 n2, l)``.  Each of the ``k`` terms is an
    orthogonal STP factorization.  ``rank`` is the uniform truncation rank
    applied to every B_i^{(j)} (TMSTP-SVD / Algorithm 8); ``None`` keeps the
    full SVD of each B_i.
    """
    L = get_transform(transform)
    A_bar = L.forward(A)
    n3 = int(A_bar.shape[2])
    if A.shape[0] != m1 * m2 or A.shape[1] != n1 * n2:
        raise ValueError(
            f"A shape {tuple(A.shape)} incompatible with "
            f"m1,m2,n1,n2 = {m1},{m2},{n1},{n2}"
        )

    U_slices: list[list[Array]] = [[] for _ in range(k)]
    S_slices: list[list[Array]] = [[] for _ in range(k)]
    V_slices: list[list[Array]] = [[] for _ in range(k)]

    for j in range(n3):
        terms = kronecker_terms(A_bar[:, :, j], m1, m2, n1, n2, k)
        for i, (B, C) in enumerate(terms):
            U, Sigma, V = _svd_factors(B, C, rank)
            U_slices[i].append(U)
            S_slices[i].append(Sigma)
            V_slices[i].append(V)
        for i in range(len(terms), k):
            raise ValueError(
                f"Requested k={k} terms but R(A[:,:,{j}]) has only {len(terms)} "
                f"singular components (max {min(m1 * n1, m2 * n2)})."
            )

    return [
        Factor(
            U=_stack_inverse(U_slices[i], L),
            S=_stack_inverse(S_slices[i], L),
            V=_stack_inverse(V_slices[i], L),
        )
        for i in range(k)
    ]


def mrstp_svd(
    A: Array,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    k: int,
    transform: str | Transform = "fft",
    rank: int | None = None,
    oversampling: int = 5,
    n_power: int = 1,
    key: Array | None = None,
) -> list[Factor]:
    """Randomized multi-term STP-SVD (Algorithm 5, Theorem 5.1).

    Sketch size is ``k + oversampling``.  ``n_power`` is the power-iteration
    count q.  The Gaussian tensor G has shape ``(m2 n2, k+s, l)`` as in
    Theorem 5.1 (Algorithm 5's printed G shape is a typo).
    """
    L = get_transform(transform)
    A_bar = L.forward(A)
    n3 = int(A_bar.shape[2])
    s = int(oversampling)
    ncols = m2 * n2
    sketch = k + s
    if sketch > min(m1 * n1, m2 * n2):
        sketch = min(m1 * n1, m2 * n2)
        s = max(0, sketch - k)
    if key is None:
        key = random.PRNGKey(0)
    G = random.normal(key, (ncols, sketch, n3), dtype=jnp.float64)
    if jnp.issubdtype(A_bar.dtype, jnp.complexfloating):
        key_i = random.fold_in(key, 1)
        G = G + 1j * random.normal(key_i, G.shape, dtype=jnp.float64)
    G_bar = L.forward(G)

    U_slices: list[list[Array]] = [[] for _ in range(k)]
    S_slices: list[list[Array]] = [[] for _ in range(k)]
    V_slices: list[list[Array]] = [[] for _ in range(k)]

    for j in range(n3):
        Rmat = rearrange(A_bar[:, :, j], m1, m2, n1, n2)
        Uk, sk, Vhk = randomized_range_svd(
            Rmat, k=k, oversampling=s, n_power=n_power, G=G_bar[:, :, j]
        )
        terms = kronecker_terms_from_svd(Uk, sk, Vhk, m1, m2, n1, n2, k)
        for i, (B, C) in enumerate(terms):
            U, Sigma, V = _svd_factors(B, C, rank)
            U_slices[i].append(U)
            S_slices[i].append(Sigma)
            V_slices[i].append(V)

    return [
        Factor(
            U=_stack_inverse(U_slices[i], L),
            S=_stack_inverse(S_slices[i], L),
            V=_stack_inverse(V_slices[i], L),
        )
        for i in range(k)
    ]


def mstp_reconstruct(
    factors: list[Factor],
    transform: str | Transform = "fft",
) -> Array:
    """Â = Σ_i U_i ⋉_L S_i ⋉_L V_i^⊤ (eq. (7))."""
    acc = None
    for fac in factors:
        term = tensor_stp(
            tensor_stp(fac.U, fac.S, transform),
            tensor_transpose(fac.V, transform),
            transform,
        )
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("no factors")
    return acc


def relative_frobenius_error(A: Array, A_hat: Array) -> Array:
    return frobenius_norm(A - A_hat) / frobenius_norm(A)


def stp_svd_matrix(
    A: Array,
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    k: int = 1,
    rank: int | None = None,
) -> list[tuple[Array, Array, Array]]:
    """Matrix multi-term STP-SVD (Algorithms 1–2 / Theorems 4.1–4.2)."""
    terms = kronecker_terms(A, m1, m2, n1, n2, k)
    out: list[tuple[Array, Array, Array]] = []
    for B, C in terms:
        out.append(_svd_factors(B, C, rank))
    return out
