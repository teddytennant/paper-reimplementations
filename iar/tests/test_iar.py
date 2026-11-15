"""Paper identities for one-atom Iterative Atom Refinement."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from iar import (
    combinatorial_codes,
    multi_restart_recovery,
    one_atom_iar,
    orthogonal_dictionary,
    refine_step,
    selection_counts,
    selection_frequencies,
    top_l_indices,
)


def test_update_identity():
    # Fixed 4 by 6 matrix, L=2. d0 = e_0, so scores are the first row:
    #   [1, 0.5, 0.2, 0.9, 0.1, 0.8]
    # Stable ascending argsort ends in indices 3 then 0 (scores 0.9, 1.0).
    # Selected columns sum to [1.9, 0, 0, 1].
    # ||sum|| = sqrt(1.9**2 + 1) = sqrt(4.61) = 2.1470910553583886
    # normalized = [1.9/sqrt(4.61), 0, 0, 1/sqrt(4.61)]
    #            = [0.884918, 0, 0, 0.465746] to 6 decimals.
    Y = jnp.array(
        [
            [1.0, 0.5, 0.2, 0.9, 0.1, 0.8],
            [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    d0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    d1, selected = refine_step(d0, Y, L=2)
    np.testing.assert_array_equal(np.asarray(selected), np.array([3, 0]))
    expected = np.array([0.884918, 0.000000, 0.000000, 0.465746])
    np.testing.assert_array_equal(np.round(np.asarray(d1), 6), expected)

    # Same tie-break as numpy mergesort on a vector with tied scores.
    tied = np.array([1.0, 2.0, 2.0, 0.5, 2.0])
    got = np.asarray(top_l_indices(jnp.asarray(tied), 2))
    np.testing.assert_array_equal(got, np.argsort(tied, kind="mergesort")[-2:])

    trajectory = one_atom_iar(Y, L=2, n_steps=4, d0=d0)
    norms = np.asarray(jnp.linalg.norm(trajectory, axis=1))
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-12)


def test_monotonicity_principle():
    # Section 3.1, display (3.2). K=7, s=2, M=C(7,2)=21, L=3, A=I.
    # d0 proportional to p_k = 7-k, strictly decreasing.
    K, s, L = 7, 2, 3
    X = combinatorial_codes(K, s)
    assert tuple(X.shape) == (K, 21)
    A = jnp.eye(K, dtype=jnp.float64)
    Y = A @ X
    p = jnp.arange(K, 0, -1, dtype=jnp.float64)
    d0 = p / jnp.linalg.norm(p)
    d1, selected = refine_step(d0, Y, L)

    # Exact support counts, not a float tolerance. Independent of selection_counts.
    chosen = [int(i) for i in np.asarray(selected)]
    assert len(chosen) == L
    counts = []
    for k in range(K):
        counts.append(sum(1 for i in chosen if float(X[k, i]) != 0.0))
    assert counts == [int(c) for c in np.asarray(selection_counts(selected, X))]
    assert all(counts[k] >= counts[k + 1] for k in range(K - 1))

    freqs = np.asarray(selection_frequencies(selected, X))
    np.testing.assert_array_equal(freqs, np.asarray(counts, dtype=np.float64) / L)

    coeffs = np.asarray(A.T @ d1)
    assert all(coeffs[k] >= coeffs[k + 1] for k in range(K - 1))


def test_amplification():
    # K=7, s=2, L=3 plateaus at sqrt(3/4) ≈ 0.866.
    # K=9, s=2, L=4 plateaus at sqrt(4/5) ≈ 0.894.
    # Both are below 0.9 for a correct top-L update. Smallest instance with
    # M/K an integer and |<d, e_0>| > 0.9 after three steps is K=11, s=2,
    # M=55, L=5, which plateaus at sqrt(5/6) ≈ 0.913.
    K, s, L = 11, 2, 5
    X = combinatorial_codes(K, s)
    assert int(X.shape[1]) == 55
    assert int(X.shape[1]) // K == L
    A = jnp.eye(K, dtype=jnp.float64)
    Y = A @ X
    p = jnp.full(K, 0.25).at[0].set(1.0)
    d0 = p / jnp.linalg.norm(p)
    trajectory = one_atom_iar(Y, L, n_steps=3, d0=d0)
    d = trajectory[3]
    align = abs(float(d[0]))
    argmax = int(jnp.argmax(jnp.abs(d)))
    assert align > 0.9
    assert argmax == 0


def test_restarts():
    K, s, L = 7, 2, 3
    X = combinatorial_codes(K, s)
    Y = jnp.eye(K, dtype=jnp.float64) @ X
    kept = np.asarray(
        multi_restart_recovery(Y, L, n_restarts=12, n_steps=3, seed=0)
    )
    for i in range(kept.shape[0]):
        for j in range(i):
            assert abs(float(np.dot(kept[i], kept[j]))) < 0.5
    recovered = {}
    for row in kept:
        k = int(np.argmax(np.abs(row)))
        align = abs(float(row[k]))
        if align > 0.8:
            recovered[k] = max(recovered.get(k, 0.0), align)
    assert len(recovered) >= 3


def test_data_helpers():
    X = np.asarray(combinatorial_codes(3, 2))
    expected = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(X, expected)

    K, s = 7, 2
    codes = np.asarray(combinatorial_codes(K, s))
    assert codes.shape == (K, 21)
    assert np.all(codes.sum(axis=0) == s)
    # Each binary s-sparse vector once.
    packed = [tuple(np.nonzero(codes[:, j])[0].tolist()) for j in range(codes.shape[1])]
    assert packed == sorted(set(packed))
    assert len(packed) == 21

    for N, K_orth, seed in ((5, 5, 0), (6, 3, 1)):
        A = orthogonal_dictionary(N, K_orth, seed)
        gram = np.asarray(A.T @ A)
        np.testing.assert_allclose(gram, np.eye(K_orth), atol=1e-12)
