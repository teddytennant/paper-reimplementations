"""The RCM objective: Eqs. 1, 2, 3 and 5 of arXiv:2609.03158.

Every function here takes V, the projected visual tokens g(f_v(X_v)) in R^{n x d}.
That space is the paper's point (Sec. 3.3, "Similarity in LLM space"): coverage is
measured where the tokens enter the decoder, after the multimodal projector, not in
the vision encoder's own feature space. Their Table 10 ablates exactly this swap and
loses 7.3 TextVQA points, so passing pre-projector features here is not the method.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "similarity_matrix",
    "centered_similarity",
    "calibrated_utilities",
    "coverage",
    "total_coverage",
]


def _as_float(x):
    """Cast to the widest float jax is configured for (float64 with x64 on)."""
    return jnp.asarray(x, dtype=jnp.result_type(float))


def similarity_matrix(V):
    """M_ij = sim(v_i, v_j), cosine similarity in projector-output space (Eq. 2, 3).

    Args:
        V: (n, d) projected visual tokens.

    Returns:
        (n, n) cosine similarity matrix, symmetric, M_ii = 1.

    The matmul is forced to highest precision so a float32 caller does not silently
    get TF32 products on an H200.
    """
    V = _as_float(V)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / jnp.where(norms > 0, norms, 1.0)
    return jnp.matmul(Vn, Vn.T, precision="highest")


def centered_similarity(M):
    """Row-wise mean-centered similarity, M~_ij = M_ij - (1/n) sum_k M_ik (Eq. 3).

    M~_ij reads as how much more similar j is to i than the average token of the
    same image. The row mean includes the diagonal, as written in the paper.
    """
    M = _as_float(M)
    return M - jnp.mean(M, axis=1, keepdims=True)


def calibrated_utilities(V=None, *, M=None):
    """A_ij = max(0, M~_ij), the calibrated coverage utilities (Eq. 3, Eq. 5).

    Pass either the token features V or a precomputed similarity matrix M.

    Clipping is what makes the utilities non-negative, which is what makes F monotone
    submodular and buys the (1 - 1/e) greedy guarantee. Centering is what gives the
    max in Eq. 1 dynamic range: raw projector-space cosines sit in a narrow band, and
    then every candidate's gain is inflated by the same near-constant amount and
    greedy cannot separate a redundant token from a novel one (Sec. 3.3). After
    centering, a below-average match provides no coverage at all.
    """
    if (V is None) == (M is None):
        raise ValueError("pass exactly one of V or M")
    if M is None:
        M = similarity_matrix(V)
    return jnp.maximum(0.0, centered_similarity(M))


def coverage(A, S):
    """cov(i, S) = max_{j in S} A_ij, the best representative token i has (Eq. 1).

    Args:
        A: (n, n) coverage utilities.
        S: indices of the retained subset. Empty S gives all-zero coverage, matching
            the c_i = 0 initialization in Eq. 6.

    Returns:
        (n,) per-token covered value.
    """
    A = _as_float(A)
    S = jnp.asarray(S, dtype=jnp.int32).reshape(-1)
    if S.size == 0:
        return jnp.zeros(A.shape[0], dtype=A.dtype)
    return jnp.max(A[:, S], axis=1)


def total_coverage(A, w, S):
    """F(S) = sum_i w_i * max_{j in S} A_ij, the RCM objective (Eq. 5).

    Args:
        A: (n, n) coverage utilities from calibrated_utilities.
        w: (n,) demand weights, w_i >= 0.
        S: indices of the retained subset.

    With A >= 0 and w >= 0 this is monotone and submodular, which is the whole reason
    the greedy selector in select.py has an approximation guarantee.
    """
    w = _as_float(w)
    return jnp.sum(w * coverage(A, S))
