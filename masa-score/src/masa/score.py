"""MASA pair, block, vertical, and slash scores.

Pair score (kept exactly):

    s(i, j) = 2 * A_ij * <O_i, X_j> - A_ij**2 * ||X_j||**2

O is the residual attention output, shape (N, d). X holds keys or values,
shape (N, d). A has shape (N, N).

The vectorized matrix is

    D = 2 * A * C - (A**2) * (1 n^T)

with C_ij = <O_i, X_j> and n_j = ||X_j||**2. C is O @ X.T. That product does
not materialize an (N, N, d) tensor.

If X @ X.T = I and the residual is the full attention output O = A @ X, then
<O_i, X_j> = A_ij and ||X_j||**2 = 1, so s(i, j) reduces to A_ij**2.

The block score is the same squared-error reduction when one query keeps
several keys at once. It equals the pair score for a single key, and it is
not the sum of pair scores when the kept keys are correlated.
"""

from masa._host import configure_host

configure_host()

import jax.numpy as jnp


def _as_triple(O, X, A):
    O = jnp.asarray(O)
    X = jnp.asarray(X)
    A = jnp.asarray(A)
    if O.ndim != 2 or X.ndim != 2 or A.ndim != 2:
        raise ValueError("O, X, and A must be rank-2 arrays")
    n, d = X.shape
    if O.shape != (n, d):
        raise ValueError(f"O shape {O.shape} does not match X shape {X.shape}")
    if A.shape != (n, n):
        raise ValueError(f"A shape {A.shape} is not ({n}, {n})")
    return O, X, A


def pair_score(O, X, A):
    """Return the pair-score matrix D, shape (N, N).

    D_ij = 2 * A_ij * <O_i, X_j> - A_ij**2 * ||X_j||**2
         = 2 * A_ij * C_ij - A_ij**2 * n_j

    C = O @ X.T and n_j = ||X_j||**2. The ones-vector outer product 1 n^T is
    the (N, N) matrix whose columns are constant and equal to n. No (N, N, d)
    tensor is formed.
    """
    O, X, A = _as_triple(O, X, A)
    C = O @ X.T
    n = jnp.sum(X * X, axis=-1)
    ones = jnp.ones((A.shape[0], 1), dtype=A.dtype)
    ones_nT = ones @ n.reshape(1, -1)
    return 2.0 * A * C - (A ** 2) * ones_nT


def block_score(O, X, A, keys):
    """Squared-error reduction when every query keeps `keys` together.

    For key set S,

        b_i = 2 <O_i, sum_{j in S} A_ij X_j> - ||sum_{j in S} A_ij X_j||**2

    which is ||O_i||**2 - ||O_i - sum_{j in S} A_ij X_j||**2. For |S| = 1 this
    is the pair score. For |S| > 1 the expansion of the squared norm contains
    cross terms 2 A_ij A_ik <X_j, X_k>, so b_i is not the sum of pair scores
    unless those inner products vanish.

    `keys` is a 1-d index array. The kept values are applied with a (N, |S|)
    by (|S|, d) product, not an (N, N, d) tensor.
    """
    O, X, A = _as_triple(O, X, A)
    keys = jnp.asarray(keys, dtype=jnp.int32)
    if keys.ndim != 1:
        raise ValueError("keys must be a 1-d index array")
    summed = A[:, keys] @ X[keys, :]
    align = jnp.sum(O * summed, axis=-1)
    sq = jnp.sum(summed * summed, axis=-1)
    return 2.0 * align - sq


def vertical_score(pair_scores):
    """Sum of pair scores over queries i, one score per key j."""
    pair_scores = jnp.asarray(pair_scores)
    if pair_scores.ndim != 2:
        raise ValueError("pair_scores must be rank-2")
    return jnp.sum(pair_scores, axis=0)


def slash_offsets(n):
    """Diagonal offsets ell = i - j, from -(n - 1) to (n - 1)."""
    return jnp.arange(-(n - 1), n)


def slash_score(pair_scores):
    """Sum of pair scores over i - j = ell.

    The returned vector has length 2N - 1. Index ell + (N - 1) holds the sum
    for offset ell.
    """
    pair_scores = jnp.asarray(pair_scores)
    if pair_scores.ndim != 2 or pair_scores.shape[0] != pair_scores.shape[1]:
        raise ValueError("slash score expects a square pair-score matrix")
    n = pair_scores.shape[0]
    sums = []
    for ell in range(-(n - 1), n):
        sums.append(jnp.trace(pair_scores, offset=-ell))
    return jnp.stack(sums)


def soft_balance(scores, eps=1e-6):
    """Divide each row by (mean absolute row score + eps).

    Practical stabilizer so rows are comparable before ranking. Not a
    transcribed paper equation. `eps` is a practical floor, not a claimed
    theorem constant.
    """
    scores = jnp.asarray(scores)
    if scores.ndim == 1:
        scale = jnp.mean(jnp.abs(scores)) + eps
        return scores / scale
    scale = jnp.mean(jnp.abs(scores), axis=-1, keepdims=True) + eps
    return scores / scale


def clip_scores(scores, multiple=10.0):
    """Cap scores at `multiple` times the median positive score.

    Upper clip only. If there is no positive score, `scores` is returned
    unchanged. `multiple` (default 10) is a practical choice, not a claimed
    theorem constant. This is a practical stabilizer, not a transcribed
    paper equation.
    """
    scores = jnp.asarray(scores)
    positive = jnp.ravel(scores)
    positive = positive[positive > 0]
    if int(positive.shape[0]) == 0:
        return scores
    cap = multiple * jnp.median(positive)
    return jnp.minimum(scores, cap)


def rank_units(scores, k, clip_multiple=10.0):
    """Keep the k units with the largest score, after clipping.

    Clipping runs before ranking. The clip multiple is a practical choice,
    not a claimed theorem constant. The mask is boolean and has the same
    shape as `scores`. Ties follow argsort order.
    """
    scores = jnp.asarray(scores)
    k = int(k)
    if k < 0:
        raise ValueError("k must be non-negative")
    clipped = clip_scores(scores, multiple=clip_multiple)
    flat = jnp.ravel(clipped)
    k = min(k, int(flat.shape[0]))
    chosen = jnp.argsort(-flat)[:k]
    mask = jnp.zeros(flat.shape, dtype=bool)
    if k > 0:
        mask = mask.at[chosen].set(True)
    return mask.reshape(scores.shape)


def reconstruction_residual(A, V, mask):
    """Frobenius norm ||A @ V - (M * A) @ V||."""
    A = jnp.asarray(A)
    V = jnp.asarray(V)
    mask = jnp.asarray(mask).astype(A.dtype)
    diff = A @ V - (mask * A) @ V
    return jnp.linalg.norm(diff)
