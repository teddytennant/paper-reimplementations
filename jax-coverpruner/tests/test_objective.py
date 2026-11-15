"""Properties of F(S), the RCM objective of Eq. 5.

Monotone and submodular is not decoration: it is the premise of the (1 - 1/e)
guarantee the paper claims in Sec. 3.2, and it is what lets lazy greedy skip
candidates without changing the answer.
"""

import itertools

import numpy as np
import reference as ref

from coverpruner import (
    calibrated_utilities,
    centered_similarity,
    coverage,
    similarity_matrix,
    total_coverage,
)


def test_utilities_match_numpy_reference():
    for seed in range(5):
        _, _, V, _ = ref.random_instance(seed, n=40, d=16)
        assert np.allclose(np.asarray(calibrated_utilities(V)), ref.utilities(V))


def test_similarity_is_cosine():
    _, _, V, _ = ref.random_instance(0, n=25, d=12)
    M = np.asarray(similarity_matrix(V))
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)
    assert M.min() >= -1 - 1e-12
    assert M.max() <= 1 + 1e-12


def test_centering_zeroes_each_row_mean():
    """Eq. 3 centers per row, so every row of M~ averages to zero."""
    _, _, V, _ = ref.random_instance(1, n=30, d=16)
    Mt = np.asarray(centered_similarity(similarity_matrix(V)))
    assert np.allclose(Mt.mean(axis=1), 0.0, atol=1e-12)


def test_utilities_non_negative_and_self_covering():
    """Clipping gives A >= 0, and A_ii is the largest entry of row i."""
    for seed in range(5):
        A, _, _, _ = ref.random_instance(seed, n=30, d=16)
        assert A.min() >= 0.0
        assert np.all(np.diag(A) >= A.max(axis=1) - 1e-12)


def test_empty_set_has_zero_coverage():
    """c_i starts at 0 in Eq. 6, so F(empty) = 0."""
    A, w, _, _ = ref.random_instance(3, n=20)
    assert np.allclose(np.asarray(coverage(A, [])), 0.0)
    assert float(total_coverage(A, w, [])) == 0.0


def test_total_coverage_matches_reference():
    A, w, _, _ = ref.random_instance(4, n=24)
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(1, 12))
        S = rng.choice(24, m, replace=False)
        assert np.isclose(float(total_coverage(A, w, S)), ref.F(A, w, S), rtol=1e-12)


def test_monotone():
    """F(S + e) >= F(S) on random instances, adding one token at a time."""
    rng = np.random.default_rng(7)
    for seed in range(25):
        A, w, _, _ = ref.random_instance(seed, n=20, d=10)
        order = rng.permutation(20)
        prev, S = 0.0, []
        for e in order:
            S.append(int(e))
            value = float(total_coverage(A, w, S))
            assert value >= prev - 1e-12
            prev = value


def test_submodular_exhaustive_small():
    """F(S + e) - F(S) >= F(T + e) - F(T) for every S subset of T, every e.

    n = 8 means 3^8 = 6561 (S, T) pairs per instance, so this is exhaustive rather
    than sampled.
    """
    n = 8
    for seed in range(4):
        A, w, _, _ = ref.random_instance(seed, n=n, d=6, clusters=3)
        for mask_t in range(1 << n):
            T = [i for i in range(n) if mask_t >> i & 1]
            sub = [i for i in range(n) if not mask_t >> i & 1]
            f_t = ref.F(A, w, T)
            for mask_s in _submasks(T):
                S = [i for i in T if i in mask_s]
                f_s = ref.F(A, w, S)
                for e in sub:
                    g_s = ref.F(A, w, S + [e]) - f_s
                    g_t = ref.F(A, w, T + [e]) - f_t
                    assert g_s >= g_t - 1e-12


def _submasks(T):
    for size in range(len(T) + 1):
        for combo in itertools.combinations(T, size):
            yield set(combo)
