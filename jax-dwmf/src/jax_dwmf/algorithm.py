"""Algorithm 1 / Theorem 16: diagonal truncation and small-block transformation.

Approximates N ≈ f(M) for M = γI + DAD (Eq. (5)) by evaluating f on the
principal block indexed by the heavy coordinates of D (Eqs. (10)–(13), (C1)–(C2)).

The block matrix function is computed by dense Hermitian eigh, not QSVT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from .truncation import truncate_diagonal

Array = jax.Array
ScalarFn = Callable[[Array], Array]


def apply_f_eigh(block: Array, f: ScalarFn) -> Array:
    """B = f(M̃_S) via M̃_S = V Λ V†, B = V f(Λ) V† (Alg. 1 steps 4–5, Eq. (11))."""
    evals, evecs = jnp.linalg.eigh(block)
    return (evecs * f(evals)) @ jnp.conjugate(evecs).T


def _hermitianize_upper(queried: Array) -> Array:
    """Enforce Hermitian symmetry on a queried upper triangle (Alg. 1 step 3)."""
    lower = jnp.conjugate(jnp.triu(queried, k=1)).T
    return jnp.triu(queried) + lower


@dataclass
class SparseMatrixFunction:
    """Support-sparse representation of N in Eq. (12) / (C2).

    Attributes
    ----------
    n : ambient dimension
    S : retained index set, ordered as {i_1, …, i_s} (0-based)
    B : s × s block f(M̃_S)
    f_gamma : f(γ), the complement-diagonal value
    d_tilde : truncated diagonal values on S
    J : row supports J_t = {i_u ∈ S : B(t, π(i_u)) ≠ 0} (Alg. 1 step 6)
    """

    n: int
    S: Array
    B: Array
    f_gamma: float
    d_tilde: Array
    J: tuple[tuple[int, ...], ...]

    @property
    def s(self) -> int:
        return int(self.S.shape[0])

    def pi(self, i: int) -> int:
        """Bijection π : S → [s] with π(i_t) = t (Alg. 1 step 2, 0-based)."""
        matches = jnp.nonzero(self.S == i)[0]
        if matches.shape[0] == 0:
            raise KeyError(i)
        return int(matches[0])

    def contains(self, i: int) -> bool:
        return bool(jnp.any(self.S == i))


def query_N(repr: SparseMatrixFunction, i: int, j: int) -> Array:
    """Sparse query Q_sp(N) after Theorem 16 / Eq. (C19)."""
    in_i = repr.contains(i)
    in_j = repr.contains(j)
    if in_i and in_j:
        return repr.B[repr.pi(i), repr.pi(j)]
    if i == j:
        return jnp.asarray(repr.f_gamma, dtype=repr.B.dtype)
    return jnp.asarray(0.0, dtype=repr.B.dtype)


def materialize_N(repr: SparseMatrixFunction) -> Array:
    """Dense n × n array of N (for small-n tests)."""
    n = repr.n
    N = jnp.zeros((n, n), dtype=repr.B.dtype)
    N = N + jnp.eye(n, dtype=repr.B.dtype) * repr.f_gamma
    if repr.s == 0:
        return N
    # Overwrite the S × S block; complement diagonal stays f(γ).
    S = repr.S
    N = N.at[S[:, None], S[None, :]].set(repr.B)
    return N


def perturbation_nu(
    A_op: float,
    D_op: float,
    D_F: float,
    eps_D: float,
    eps_A: float,
) -> float:
    """ν in Eq. (15) / (C10): 2||A||_op ||D||_op ϵ_D + ϵ_A ||D||_F^2."""
    return 2.0 * A_op * D_op * eps_D + eps_A * (D_F**2)


def diagonal_truncation_transform(
    key: jax.Array,
    d: Array,
    A: Array,
    gamma: float,
    f: ScalarFn,
    *,
    eps_D: float,
    delta_D: float = 1e-6,
    eps_A: float = 0.0,
    delta_A: float = 0.0,
    exact_support: bool = False,
    query_A: Optional[Callable[[int, int], Array]] = None,
) -> SparseMatrixFunction:
    """Algorithm 1.

    Parameters
    ----------
    d : nonnegative diagonal of D
    A : Hermitian n × n factor (used when ``query_A`` is None)
    gamma : shift γ in M = γI + DAD
    f : scalar function, applied entrywise to eigenvalues of the block
    eps_D, delta_D : Lemma 15 truncation parameters
    eps_A, delta_A : noisy query Q_{ϵ_A, δ_A}(A); ϵ_A = 0 is exact Q(A)
    exact_support : if True, take S = H_{ϵ_D} (successful event of Lemma 15)
    query_A : optional entry oracle; defaults to reading ``A``
    """
    d = jnp.asarray(d)
    n = int(d.shape[0])
    S, d_S = truncate_diagonal(
        key, d, eps_D, delta_D, exact=exact_support
    )
    s = int(S.shape[0])
    f_gamma = float(jnp.asarray(f(jnp.asarray(gamma, dtype=d.dtype))))

    if s == 0:
        return SparseMatrixFunction(
            n=n,
            S=S,
            B=jnp.zeros((0, 0), dtype=d.dtype),
            f_gamma=f_gamma,
            d_tilde=d_S,
            J=tuple(),
        )

    if query_A is None:
        A = jnp.asarray(A)

        def query_A(i: int, j: int, _A=A) -> Array:  # type: ignore[misc]
            return _A[i, j]

    # Alg. 1 step 3 / Eq. (10), (C1): query upper triangle, Hermitianize.
    queried = jnp.zeros((s, s), dtype=jnp.result_type(d.dtype, jnp.float64))
    S_list: Sequence[int] = [int(i) for i in S]
    noise_key = jax.random.fold_in(key, 17)
    n_queries = s * (s + 1) // 2
    noise_keys = jax.random.split(noise_key, max(n_queries, 1))
    q = 0
    for t, it in enumerate(S_list):
        for u in range(t, s):
            iu = S_list[u]
            entry = jnp.asarray(query_A(it, iu))
            if eps_A > 0.0:
                # Definition 11: additive ϵ_A-accurate entry (here: exact
                # entry plus a bounded perturbation of magnitude ≤ ϵ_A).
                # δ_A is a failure probability in the paper; tests use ϵ_A = 0.
                noise = jax.random.uniform(
                    noise_keys[q], minval=-eps_A, maxval=eps_A
                )
                entry = entry + noise.astype(entry.dtype)
            queried = queried.at[t, u].set(entry)
            q += 1
    A_S = _hermitianize_upper(queried)
    Ds = d_S.astype(A_S.dtype)
    # M̃_S(t, u) = γ δ_{t,u} + d̃_{i_t} A(i_t, i_u) d̃_{i_u}
    M_S = gamma * jnp.eye(s, dtype=A_S.dtype) + (Ds[:, None] * A_S * Ds[None, :])
    B = apply_f_eigh(M_S, f)

    # Alg. 1 step 6: row supports of B inside S.
    zero_tol = 0.0
    J = []
    for t in range(s):
        cols = tuple(
            S_list[u] for u in range(s) if abs(complex(B[t, u])) > zero_tol
        )
        J.append(cols)

    return SparseMatrixFunction(
        n=n,
        S=S,
        B=B,
        f_gamma=f_gamma,
        d_tilde=Ds,
        J=tuple(J),
    )


def true_M(d: Array, A: Array, gamma: float) -> Array:
    """M = γI + DAD (Eq. (5))."""
    D = jnp.diag(d)
    n = d.shape[0]
    return gamma * jnp.eye(n, dtype=A.dtype) + D @ A @ D


def true_f_M(d: Array, A: Array, gamma: float, f: ScalarFn) -> Array:
    """Exact f(M) by dense eigh of the full matrix (small-n oracle)."""
    return apply_f_eigh(true_M(d, A, gamma), f)


def operator_norm(M: Array) -> Array:
    """||·||_op induced by the Euclidean norm, for Hermitian M."""
    return jnp.max(jnp.abs(jnp.linalg.eigvalsh(0.5 * (M + jnp.conjugate(M).T))))
