"""Combinatorial codes and orthogonal dictionaries for the IAR model."""

import itertools
import math

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def combinatorial_codes(K, s):
    """Binary s-sparse codes, each support once, in lexicographic order.

    Returns X of shape (K, M) with M = C(K, s). Column j is the indicator of
    the j-th tuple from ``itertools.combinations(range(K), s)``.
    """
    K = int(K)
    s = int(s)
    if K < 0 or not 0 <= s <= K:
        raise ValueError(f"need 0 <= s <= K, got K={K}, s={s}")
    M = math.comb(K, s)
    X = np.zeros((K, M), dtype=np.float64)
    for j, comb in enumerate(itertools.combinations(range(K), s)):
        X[list(comb), j] = 1.0
    return jnp.asarray(X)


def orthogonal_dictionary(N, K, seed):
    """N by K matrix A with A.T @ A = I.

    Columns are the thin QR factor of an i.i.d. standard Gaussian matrix, with
    the sign of each column chosen so that the diagonal of R is positive. When
    N = K this is a Haar-distributed orthogonal matrix.
    """
    N = int(N)
    K = int(K)
    if K > N:
        raise ValueError(f"need N >= K for orthonormal columns, got N={N}, K={K}")
    if K <= 0 or N <= 0:
        raise ValueError(f"need positive N, K, got N={N}, K={K}")
    key = jax.random.PRNGKey(int(seed))
    gaussian = jax.random.normal(key, (N, K), dtype=jnp.float64)
    Q, R = jnp.linalg.qr(gaussian, mode="reduced")
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1.0, signs)
    return Q * signs
