"""Checks for the linear-CKA effective-depth diagnostic."""

import numpy as np
import pytest

from effective_depth import (
    cka,
    deff,
    deff_from_rho,
    deff_on_updates,
    drop_top_pcs,
    first_non_bos,
    mean_pool,
    orthogonal_update_population_rho,
    token_normed,
)


def _ref_cka(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = np.sum((B.T @ A) ** 2)
    den = np.linalg.norm(A.T @ A, ord="fro") * np.linalg.norm(B.T @ B, ord="fro")
    if den == 0.0:
        return 0.0
    return float(num / den)


def _helmert_columns(n):
    """n-1 integer contrasts, orthogonal to each other and to the all-ones vector."""
    cols = []
    for j in range(1, n):
        v = np.zeros(n, dtype=np.float64)
        v[:j] = 1.0
        v[j] = -float(j)
        cols.append(v.reshape(-1, 1))
    return cols


def test_identical_layers_deff_is_one():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 4))
    layers = [base.copy() for _ in range(5)]
    out = deff(layers)
    assert out["rho"].shape == (4,)
    assert out["deff"] == pytest.approx(1.0, abs=1e-12)
    assert cka(base, base) == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_layers_deff_equals_L():
    L = 5
    layers = _helmert_columns(L + 1)[:L]
    out = deff(layers)
    assert out["rho"] == pytest.approx(np.zeros(L - 1), abs=1e-12)
    assert out["deff"] == pytest.approx(float(L), abs=1e-12)
    for i in range(L):
        for j in range(i + 1, L):
            assert cka(layers[i], layers[j]) == pytest.approx(0.0, abs=1e-12)


def test_f_l_matches_closed_form():
    for L in range(1, 33):
        rho = np.zeros(L - 1, dtype=np.float64)
        out = deff_from_rho(rho, L)
        assert out["f_l"] == pytest.approx(2.0 * L / (L + 1), abs=0.0, rel=0.0)
        assert out["rho"].shape == (L - 1,)


def test_population_orthogonal_updates_match_f_l():
    for L in (2, 4, 8, 16, 32):
        rho = orthogonal_update_population_rho(L)
        assert rho.shape == (L - 1,)
        # Spot-check the 1-based sum for k=1.
        k = 1
        manual = sum(i / (i + k) for i in range(1, L - k + 1)) / (L - k)
        assert rho[k - 1] == pytest.approx(manual, abs=1e-15)
        out = deff_from_rho(rho, L)
        assert out["deff"] == pytest.approx(2.0 * L / (L + 1), abs=1e-9)


def test_l2_hand_computed_cka_and_deff():
    H = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, -1.0, 3.0],
            [2.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    H2 = np.array(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.0, 0.5],
            [-1.0, 2.0, 1.0],
            [0.25, 0.5, -0.5],
        ],
        dtype=np.float64,
    )
    c_ref = _ref_cka(H, H2)
    # L=2: D_eff = 2 / (1 + 2 * (1 - 1/2) * rho) = 2 / (1 + rho)
    d_ref = 2.0 / (1.0 + c_ref)
    f_ref = 2.0 * 2 / (2 + 1)
    gap_ref = (f_ref - d_ref) / f_ref

    assert cka(H, H2) == pytest.approx(c_ref, abs=1e-12)
    out = deff([H, H2])
    assert out["rho"].shape == (1,)
    assert out["rho"][0] == pytest.approx(c_ref, abs=1e-12)
    assert out["deff"] == pytest.approx(d_ref, abs=1e-12)
    assert out["f_l"] == pytest.approx(f_ref, abs=1e-15)
    assert out["gap"] == pytest.approx(gap_ref, abs=1e-12)

    stacked = deff(np.stack([H, H2], axis=0))
    assert stacked["deff"] == pytest.approx(d_ref, abs=1e-12)


def test_token_normed_changes_deff_when_norms_differ():
    rng = np.random.default_rng(1)
    n, seq, d, L = 32, 2, 8, 3
    signal = rng.normal(size=(n, d))
    signal = signal / np.linalg.norm(signal, axis=1, keepdims=True)
    token_layers = []
    for _ in range(L):
        tokens = np.zeros((n, seq, d), dtype=np.float64)
        tokens[:, 0, :] = 0.01 * signal
        noise = rng.normal(size=(n, d))
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        tokens[:, 1, :] = 100.0 * noise
        norms = np.linalg.norm(tokens, axis=-1)
        assert norms.max() > 10.0 * norms.min()
        token_layers.append(tokens)

    d_mean = deff([mean_pool(t) for t in token_layers])["deff"]
    d_norm = deff([token_normed(t) for t in token_layers])["deff"]
    assert abs(d_mean - d_norm) > 1e-3


def test_drop_top_pcs_removes_planted_rank1():
    rng = np.random.default_rng(2)
    n, d = 40, 7
    v = rng.normal(size=d)
    v = v / np.linalg.norm(v)
    scores = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=(n, d))
    offset = rng.normal(size=d)
    X = scores[:, None] * v[None, :] + noise + offset
    residual = drop_top_pcs(X, 1)
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    top = vh[0]
    assert abs(np.dot(top, v)) > 0.99
    assert residual @ top == pytest.approx(np.zeros(n), abs=1e-8)
    # Noise tilts the top PC off v, but the planted direction is mostly gone.
    before = np.linalg.norm(centered @ v)
    after = np.linalg.norm(residual @ v)
    assert after < 0.05 * before


def test_zero_denominator_cka_is_zero():
    constant = np.ones((6, 3), dtype=np.float64)
    assert cka(constant, constant) == 0.0
    assert cka(constant, np.arange(18, dtype=np.float64).reshape(6, 3)) == 0.0


def test_first_non_bos_takes_index_one():
    X = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert first_non_bos(X) == pytest.approx(X[:, 1, :])


def test_deff_on_updates_matches_manual_differences():
    rng = np.random.default_rng(3)
    H1 = rng.normal(size=(6, 3))
    H2 = H1 + rng.normal(size=(6, 3))
    H3 = H2 + rng.normal(size=(6, 3))
    out = deff_on_updates([H1, H2, H3])
    manual = deff([H1, H2 - H1, H3 - H2])
    assert out["deff"] == pytest.approx(manual["deff"], abs=1e-12)
    assert out["rho"] == pytest.approx(manual["rho"], abs=1e-12)
