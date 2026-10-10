"""Identity tests for MASA scores. These assert formulas, not imports."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from masa import (
    block_score,
    clip_scores,
    pair_score,
    rank_units,
    reconstruction_residual,
    slash_offsets,
    slash_score,
    soft_balance,
    vertical_score,
)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _naive_pair(O, X, A):
    n = A.shape[0]
    out = jnp.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            inner = jnp.dot(O[i], X[j])
            xnorm = jnp.dot(X[j], X[j])
            out = out.at[i, j].set(2.0 * A[i, j] * inner - (A[i, j] ** 2) * xnorm)
    return out


def test_pair_score_matches_definition():
    O = jnp.array(
        [[0.4, -1.2, 0.3], [1.5, 0.2, -0.7], [-0.3, 0.8, 0.1]],
        dtype=jnp.float64,
    )
    X = jnp.array(
        [[0.5, 0.0, 1.0], [-0.2, 0.7, 0.4], [1.1, -0.3, 0.2]],
        dtype=jnp.float64,
    )
    A = jnp.array(
        [[0.6, -1.4, 0.2], [1.0, 0.3, -0.8], [0.0, 2.0, -0.5]],
        dtype=jnp.float64,
    )
    got = pair_score(O, X, A)
    assert got.shape == (3, 3)
    assert jnp.allclose(got, _naive_pair(O, X, A), atol=1e-12)


def test_score_matrix_equals_pair_scores_without_rank3():
    """D = 2*A*C - (A**2)*(1 n^T) equals the scalar pair-score matrix."""
    O = jnp.array([[1.0, 2.0], [-0.5, 0.25], [0.3, -1.0]], dtype=jnp.float64)
    X = jnp.array([[0.4, -0.2], [1.5, 0.5], [-0.3, 0.8]], dtype=jnp.float64)
    A = jnp.array(
        [[1.2, -0.4, 0.7], [0.3, 2.0, -1.1], [-0.6, 0.5, 0.9]],
        dtype=jnp.float64,
    )
    C = O @ X.T
    n = jnp.sum(X * X, axis=-1)
    ones_nT = jnp.ones((3, 1), dtype=jnp.float64) @ n.reshape(1, -1)
    D = 2.0 * A * C - (A ** 2) * ones_nT
    assert jnp.allclose(pair_score(O, X, A), D, atol=1e-12)
    assert jnp.allclose(D, _naive_pair(O, X, A), atol=1e-12)
    assert C.ndim == 2 and ones_nT.ndim == 2


def test_orthonormal_rows_reduce_to_a_squared():
    # Rows of X are orthonormal, and O is the full attention output A @ X.
    angle = 0.3
    c, s = jnp.cos(angle), jnp.sin(angle)
    X = jnp.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    assert jnp.allclose(X @ X.T, jnp.eye(3, dtype=jnp.float64), atol=1e-12)
    A = jnp.array(
        [[1.5, -0.7, 0.2], [0.4, 2.0, -1.1], [-0.3, 0.5, 0.8]],
        dtype=jnp.float64,
    )
    O = A @ X
    got = pair_score(O, X, A)
    assert jnp.allclose(got, A ** 2, atol=1e-10)


def test_block_score_matches_joint_reduction_not_pair_sum():
    O = jnp.array(
        [[1.0, 0.25], [-0.5, 0.75], [0.2, -0.1]],
        dtype=jnp.float64,
    )
    X = jnp.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=jnp.float64)
    A = jnp.array(
        [[1.5, -0.7, 0.2], [0.4, 1.1, -0.3], [0.2, 0.5, -0.4]],
        dtype=jnp.float64,
    )
    keys = jnp.array([0, 1])
    got = block_score(O, X, A, keys)

    contrib = A[:, 0:1] * X[0] + A[:, 1:2] * X[1]
    explicit = jnp.sum(O * O, axis=-1) - jnp.sum((O - contrib) ** 2, axis=-1)
    pair_sum = pair_score(O, X, A)[:, keys].sum(axis=-1)
    cross = -2.0 * A[:, 0] * A[:, 1] * jnp.dot(X[0], X[1])

    assert jnp.allclose(got, explicit, atol=1e-12)
    assert jnp.allclose(got - pair_sum, cross, atol=1e-12)
    assert jnp.all(jnp.abs(got - pair_sum) > 1e-6)
    assert float(jnp.dot(X[0], X[1])) != 0.0


def test_block_score_matches_pair_for_one_key():
    O = jnp.array(
        [[0.2, -0.4, 1.0], [0.5, 0.5, -0.2], [-0.1, 0.3, 0.7]],
        dtype=jnp.float64,
    )
    X = jnp.array(
        [[1.0, 0.0, 0.0], [0.2, -0.3, 0.4], [-1.0, 0.5, 0.25]],
        dtype=jnp.float64,
    )
    A = jnp.array(
        [[0.8, -1.2, 0.3], [1.4, 0.6, -0.9], [0.1, -0.4, 0.5]],
        dtype=jnp.float64,
    )
    for j in range(3):
        got = block_score(O, X, A, jnp.array([j]))
        assert jnp.allclose(got, pair_score(O, X, A)[:, j], atol=1e-12)


def test_vertical_and_slash_are_sums_of_pair_scores():
    S = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=jnp.float64,
    )
    assert jnp.allclose(vertical_score(S), jnp.array([12.0, 15.0, 18.0]))
    n = S.shape[0]
    expected = []
    for ell in range(-(n - 1), n):
        total = 0.0
        for i in range(n):
            j = i - ell
            if 0 <= j < n:
                total += float(S[i, j])
        expected.append(total)
    assert jnp.allclose(slash_offsets(n), jnp.arange(-(n - 1), n))
    assert jnp.allclose(slash_score(S), jnp.array(expected, dtype=jnp.float64))


def test_soft_balance_and_clip_are_the_practical_stabilizers():
    S = jnp.array([[1.0, -1.0, 0.0], [4.0, 0.0, 0.0]], dtype=jnp.float64)
    eps = 1e-6
    balanced = soft_balance(S, eps=eps)
    scale = jnp.mean(jnp.abs(S), axis=-1, keepdims=True) + eps
    assert jnp.allclose(balanced, S / scale)

    raw = jnp.array([1.0, -4.0, 3.0, 100.0], dtype=jnp.float64)
    multiple = 10.0
    clipped = clip_scores(raw, multiple=multiple)
    cap = multiple * jnp.median(raw[raw > 0])
    assert jnp.allclose(clipped, jnp.minimum(raw, cap))
    assert float(clipped[1]) == -4.0
    assert float(cap) == 30.0


def test_low_mass_column_beats_high_mass_at_same_budget():
    """A cancelled high-mass column loses to a low-mass aligned column.

    Columns 0 and 1 have large A**2 mass but opposite values, so they cancel
    in A @ V. Column 2 has small A**2 mass and carries the residual. At the
    same entry budget, ranking by the MASA pair score keeps that low-mass
    column and leaves a smaller residual than keeping the largest A**2 entries.
    """
    X = jnp.array([[1.0], [-1.0], [1.0]], dtype=jnp.float64)
    A = jnp.array(
        [[5.0, 5.0, 0.5], [5.0, 5.0, 0.5], [5.0, 5.0, 0.5]],
        dtype=jnp.float64,
    )
    V = X
    O = A @ V
    scores = pair_score(O, X, A)

    mass = jnp.sum(A ** 2, axis=0)
    vertical = vertical_score(scores)
    low = int(jnp.argmin(mass))
    high = int(jnp.argmax(mass))
    assert low == 2
    assert high != low
    assert float(mass[low]) < float(mass[high])
    assert float(vertical[low]) > float(vertical[high])
    assert float(vertical[low]) > float(jnp.max(vertical[jnp.arange(3) != low]))

    k = 3
    masa_mask = rank_units(scores, k)
    a2_mask = rank_units(A ** 2, k, clip_multiple=1.0e12)
    assert int(jnp.sum(masa_mask)) == k
    assert int(jnp.sum(a2_mask)) == k
    assert jnp.array_equal(masa_mask[:, low], jnp.ones(3, dtype=bool))
    assert not bool(jnp.any(a2_mask[:, low]))

    masa_residual = reconstruction_residual(A, V, masa_mask)
    a2_residual = reconstruction_residual(A, V, a2_mask)
    assert float(masa_residual) < float(a2_residual)
    assert float(masa_residual) < 1e-9

    # Same claim at column budget 1: keep the best vertical unit versus the
    # highest A**2 column. Entry budget above is the comparison against the
    # k largest A**2 entries.
    masa_col = jnp.zeros_like(A, dtype=bool).at[:, int(jnp.argmax(vertical))].set(True)
    a2_col = jnp.zeros_like(A, dtype=bool).at[:, high].set(True)
    assert float(reconstruction_residual(A, V, masa_col)) < float(
        reconstruction_residual(A, V, a2_col)
    )
