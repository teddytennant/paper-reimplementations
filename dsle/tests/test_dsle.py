"""Checks for Algorithms 1--3 against the identities in arXiv:2609.23668."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dsle.estimate import (
    dsle,
    laplacian_from_covariance,
    laplacian_from_resistance,
    maximum_spanning_tree,
    mds_plus,
    resistance_from_covariance,
    sample_covariance,
    sparsify,
)


def path_laplacian(p: int) -> jnp.ndarray:
    L = jnp.zeros((p, p), dtype=jnp.float64)
    idx = jnp.arange(p - 1)
    L = L.at[idx, idx].add(1.0)
    L = L.at[idx + 1, idx + 1].add(1.0)
    L = L.at[idx, idx + 1].set(-1.0)
    L = L.at[idx + 1, idx].set(-1.0)
    return L


def laplacian_pinv(L: jnp.ndarray) -> jnp.ndarray:
    """Pseudoinverse on the subspace orthogonal to 1."""
    evals, evecs = jnp.linalg.eigh(0.5 * (L + L.T))
    inv = jnp.where(evals > 1e-8, 1.0 / evals, 0.0)
    S = (evecs * inv) @ evecs.T
    p = L.shape[0]
    J = jnp.eye(p, dtype=jnp.float64) - jnp.ones((p, p), dtype=jnp.float64) / p
    return J @ S @ J


def sample_igmrf(L: jnp.ndarray, n: int, seed: int) -> jnp.ndarray:
    """Rows are draws from N(0, L^+); the constant mode is zeroed."""
    evals, evecs = jnp.linalg.eigh(0.5 * (L + L.T))
    inv_sqrt = jnp.where(evals > 1e-8, jnp.sqrt(1.0 / evals), 0.0)
    z = jax.random.normal(jax.random.PRNGKey(seed), (n, L.shape[0]), dtype=jnp.float64)
    return (z * inv_sqrt) @ evecs.T


def rel_err(got: jnp.ndarray, truth: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(got - truth) / jnp.linalg.norm(truth))


def path_edges(p: int) -> set[tuple[int, int]]:
    return {(i, i + 1) for i in range(p - 1)}


def offdiag_edges(L: jnp.ndarray, tol: float = 1e-8) -> set[tuple[int, int]]:
    p = L.shape[0]
    scale = float(jnp.max(jnp.abs(L)))
    cutoff = tol * max(scale, 1.0)
    found: set[tuple[int, int]] = set()
    for i in range(p):
        for j in range(i + 1, p):
            if abs(float(L[i, j])) > cutoff:
                found.add((i, j))
    return found


def fscore(estimated: set[tuple[int, int]], truth: set[tuple[int, int]]) -> float:
    tp = len(estimated & truth)
    fp = len(estimated - truth)
    fn = len(truth - estimated)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def assert_signed_psd_laplacian(L: jnp.ndarray, zero_tol: float = 1e-6) -> None:
    p = L.shape[0]
    assert L.shape == (p, p)
    assert float(jnp.max(jnp.abs(L - L.T))) <= 1e-8
    ones = jnp.ones(p, dtype=L.dtype)
    assert float(jnp.linalg.norm(L @ ones)) <= 1e-6
    evals = jnp.linalg.eigvalsh(0.5 * (L + L.T))
    assert float(evals.min()) >= -1e-6
    # Kernel of a connected signed Laplacian is span{1}. The algebraic zero
    # sits near 1e-14; the Fiedler value of these scaled path-like matrices
    # is orders of magnitude larger, so 1e-5 separates them.
    n_near = int(jnp.sum(jnp.abs(evals) <= 1e-5))
    assert n_near == 1, f"near-zero eigenvalues: {evals[:3]}"
    assert abs(float(evals[0])) <= zero_tol
    assert float(evals[1]) > 1e-5


def test_resistance_round_trip():
    p = 12
    L = path_laplacian(p)
    S = laplacian_pinv(L)
    ones = jnp.ones(p, dtype=jnp.float64)
    assert float(jnp.linalg.norm(S @ ones)) <= 1e-8
    assert rel_err(L @ S @ L, L) <= 1e-8

    R = resistance_from_covariance(S)
    assert float(jnp.max(jnp.abs(jnp.diag(R)))) == 0.0
    assert float(jnp.min(R)) >= -1e-12

    L_hat = laplacian_from_resistance(R)
    err = rel_err(L_hat, L)
    assert err <= 1e-6, err


def test_proposition_4_invariants():
    n, p = 8, 12
    X = jax.random.normal(jax.random.PRNGKey(0), (n, p), dtype=jnp.float64)
    cols = [X[:, j] for j in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            assert float(jnp.linalg.norm(cols[i] - cols[j])) > 0.0

    S = sample_covariance(X)
    R = resistance_from_covariance(S)
    # R_ij is the mean squared column difference, divided by n not n-1.
    expected = float(jnp.sum((X[:, 0] - X[:, 1]) ** 2) / n)
    assert abs(float(R[0, 1]) - expected) <= 1e-10
    assert float(jnp.min(R)) >= -1e-10

    for gamma in (1.5, 2.0):
        L = dsle(X, gamma)
        assert rel_err(L, laplacian_from_covariance(S, gamma)) <= 1e-12
        assert_signed_psd_laplacian(L)


def test_population_recovery():
    p = 12
    L = path_laplacian(p)
    S = laplacian_pinv(L)
    R = resistance_from_covariance(S)
    truth = path_edges(p)

    # MDS+ is a no-op on the exact effective-resistance EDM.
    D = mds_plus(R)
    assert rel_err(D, R) <= 1e-6, rel_err(D, R)

    L_gamma1 = laplacian_from_covariance(S, 1.0)
    err = rel_err(L_gamma1, L)
    assert err <= 1e-6, err

    L_gamma = laplacian_from_covariance(S, 1.5)
    assert_signed_psd_laplacian(L_gamma)

    abs_off = jnp.abs(L_gamma)
    abs_off = abs_off.at[jnp.diag_indices(p)].set(-1.0)
    top = jnp.argsort(abs_off.ravel())[-(p - 1) :]
    n_true = 0
    for flat in top:
        i, j = divmod(int(flat), p)
        if abs(i - j) == 1:
            n_true += 1
    assert n_true >= p - 2, n_true


def test_sparsify_population():
    p = 12
    M = 6
    L = path_laplacian(p)
    S = laplacian_pinv(L)
    L_bar = laplacian_from_covariance(S, 1.5)
    laps = sparsify(L_bar, S, M)
    assert len(laps) == M

    truth = path_edges(p)
    counts = []
    for Lm in laps:
        assert_signed_psd_laplacian(Lm)
        tr = float(jnp.trace(Lm @ S))
        assert abs(tr - (p - 1)) <= 1e-5, tr
        off = Lm.at[jnp.diag_indices(p)].set(0.0)
        counts.append(int(jnp.sum(jnp.abs(off) > 1e-8)))

    for earlier, later in zip(counts, counts[1:]):
        assert later <= earlier, counts

    edges_m1 = offdiag_edges(laps[0])
    assert truth <= edges_m1


def test_mst_keeps_negative_bridge_and_breaks_ties():
    W = jnp.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, -0.5],
            [-1.0, -0.5, 0.0],
        ],
        dtype=jnp.float64,
    )
    mask = maximum_spanning_tree(W)
    assert bool(mask[0, 1].item()) and bool(mask[1, 2].item())
    assert not bool(mask[0, 2].item())

    tied = jnp.ones((3, 3), dtype=jnp.float64) - jnp.eye(3, dtype=jnp.float64)
    tied_mask = maximum_spanning_tree(tied)
    # Equal weights: lexicographic Kruskal keeps (0, 1) and (0, 2).
    assert bool(tied_mask[0, 1].item()) and bool(tied_mask[0, 2].item())
    assert not bool(tied_mask[1, 2].item())


def _fscore_seed(seed: int, p: int = 24, n: int = 12, gamma: float = 1.5, M: int = 8) -> float:
    L = path_laplacian(p)
    X = sample_igmrf(L, n, seed)
    # Columns of a continuous draw are distinct; guard the precondition anyway.
    for i in range(p):
        for j in range(i + 1, p):
            assert float(jnp.linalg.norm(X[:, i] - X[:, j])) > 0.0
    S = sample_covariance(X)
    laps = sparsify(dsle(X, gamma), S, M)
    counts = [len(offdiag_edges(Lm)) for Lm in laps]
    target = p - 1
    m = min(range(M), key=lambda k: (abs(counts[k] - target), k))
    return fscore(offdiag_edges(laps[m]), path_edges(p))


def test_underdetermined_samples():
    seed0 = _fscore_seed(0)
    if seed0 > 0.5:
        return
    scores = [_fscore_seed(s) for s in range(5)]
    mean = sum(scores) / len(scores)
    assert mean > 0.5, f"seed0={seed0}, scores={scores}, mean={mean}"
