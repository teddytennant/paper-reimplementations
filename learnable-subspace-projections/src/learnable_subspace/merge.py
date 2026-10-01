"""Merge an orthonormal projector into low-rank factors A and B."""

import jax.numpy as jnp


def orthogonal_complement(U):
    """Orthonormal basis of the orthogonal complement of span(U).

    U is (d, k) with full column rank. Returns U_perp of shape (d, d - k).
    """
    d, k = U.shape
    if k == 0:
        return jnp.eye(d, dtype=U.dtype)
    left, _s, _vh = jnp.linalg.svd(U, full_matrices=True)
    return left[:, k:]


def merge_input(W, U):
    """Input-side merge at alpha = 1, m = 1.

    U_perp spans the complement of span(U), r = din - k.
    A = U_perp.T in R^{r x din}, B = W @ U_perp in R^{dout x r}.
    B @ A equals W @ P with P = I - U @ U.T.
    """
    U_perp = orthogonal_complement(U)
    A = U_perp.T
    B = W @ U_perp
    return A, B


def merge_output(W, U):
    """Output-side factors so B @ A = P @ W.

    P @ W = U_perp @ (U_perp.T @ W). B = U_perp, A = U_perp.T @ W.
    """
    U_perp = orthogonal_complement(U)
    A = U_perp.T @ W
    B = U_perp
    return A, B


def merge_tied_input(weights, U):
    """Layers that share an input share one U and therefore one A.

    weights is a sequence of (dout_i, din) matrices. Returns the shared A
    and a list of B_i = W_i @ U_perp.
    """
    U_perp = orthogonal_complement(U)
    A = U_perp.T
    Bs = [W @ U_perp for W in weights]
    return A, Bs
