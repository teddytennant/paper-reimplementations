"""Numerical tests for the MDL geometric controller (arXiv:2609.22043)."""

from __future__ import annotations

import numpy as np
import pytest

from mdl import (
    D,
    EPS_C,
    GAINS,
    NMAX,
    NMIN,
    SIGMA,
    SUBSPACE_DIMS,
    THETA,
    action_from_cfinal,
    build_subspaces,
    confidence_consistency,
    conflict_rate,
    decide,
    decide_from_memories,
    extract_signals,
    gated_confidence,
    pair_conflict,
    subspace_weights,
    value_encode,
)


# ---------------------------------------------------------------------------
# 1. Value encoding
# ---------------------------------------------------------------------------


def test_value_encoding_finite_and_monotonic():
    v0 = np.asarray(value_encode(0.0))
    v1 = np.asarray(value_encode(1.0))
    assert v0.shape == (D,)
    assert v1.shape == (D,)
    assert np.isfinite(v0).all()
    assert np.isfinite(v1).all()

    grid = np.linspace(0.0, 1.0, 11)
    norms = np.array([float(np.linalg.norm(value_encode(s))) for s in grid])
    diffs = np.diff(norms)
    assert np.all(diffs > 0.0), f"||v(s)|| not increasing on grid: {norms}"


def test_value_encoding_centers_and_sigma():
    # v(s) = s * 1_D + RBF peaks at c_i = (i-1)/(D-1), sigma=0.20
    s = 0.4
    v = np.asarray(value_encode(s))
    centers = np.arange(D) / (D - 1)
    expected = s + np.exp(-((s - centers) ** 2) / (2.0 * SIGMA**2))
    np.testing.assert_allclose(v, expected, atol=1e-12)
    assert SIGMA == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 2. QR projectors
# ---------------------------------------------------------------------------


def test_qr_projectors_lemma1():
    sub = build_subspaces(seed=42)
    Pis = [np.asarray(sub.Pi_wm), np.asarray(sub.Pi_r), np.asarray(sub.Pi_a)]
    I = np.eye(D)

    for Pi in Pis:
        np.testing.assert_allclose(Pi @ Pi, Pi, atol=1e-10)

    np.testing.assert_allclose(Pis[0] @ Pis[1], np.zeros((D, D)), atol=1e-10)
    np.testing.assert_allclose(Pis[0] @ Pis[2], np.zeros((D, D)), atol=1e-10)
    np.testing.assert_allclose(Pis[1] @ Pis[2], np.zeros((D, D)), atol=1e-10)

    np.testing.assert_allclose(sum(Pis), I, atol=1e-10)

    rng = np.random.default_rng(0)
    v = rng.standard_normal(D)
    energy = sum(float(np.linalg.norm(Pi @ v) ** 2) for Pi in Pis)
    np.testing.assert_allclose(energy, float(np.linalg.norm(v) ** 2), atol=1e-10)

    assert SUBSPACE_DIMS == (5, 5, 6)
    assert np.asarray(sub.P_wm).shape == (D, 5)
    assert np.asarray(sub.P_r).shape == (D, 5)
    assert np.asarray(sub.P_a).shape == (D, 6)


# ---------------------------------------------------------------------------
# 3. Wk principal gain / leakage
# ---------------------------------------------------------------------------


def test_wk_principal_gain_and_leakage():
    sub = build_subspaces(seed=42)
    weights = subspace_weights(sub)
    sk = {"wm": GAINS[0], "r": GAINS[1], "a": GAINS[2]}
    Pk = {"wm": np.asarray(sub.P_wm), "r": np.asarray(sub.P_r), "a": np.asarray(sub.P_a)}
    Wk = {k: np.asarray(weights[k]) for k in ("wm", "r", "a")}

    assert GAINS == (1.0, 0.7, 0.5)
    assert EPS_C == pytest.approx(0.10)

    for k, P in Pk.items():
        # columns of Pk live in subspace k: Wk P ≈ sk P
        np.testing.assert_allclose(Wk[k] @ P, sk[k] * P, atol=1e-10)
        for j, Pj in Pk.items():
            if j == k:
                continue
            np.testing.assert_allclose(Wk[k] @ Pj, EPS_C * Pj, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Risk inversion
# ---------------------------------------------------------------------------


def test_risk_inversion_shrinks_va_and_gA():
    high_A = decide(0.9, 0.9, 0.85)
    low_A = decide(0.9, 0.9, 0.20)
    assert high_A.va_norm < low_A.va_norm
    assert high_A.gA < low_A.gA


# ---------------------------------------------------------------------------
# 5. High-risk suppression
# ---------------------------------------------------------------------------


def test_high_risk_suppression():
    high_A = decide(0.9, 0.9, 0.85)
    low_A = decide(0.9, 0.9, 0.20)
    assert high_A.Cfinal < low_A.Cfinal
    assert high_A.action in ("Silent", "Opt-Out")
    assert high_A.adopt is False


# ---------------------------------------------------------------------------
# 6. Low relevance is not Active
# ---------------------------------------------------------------------------


def test_low_relevance_not_active():
    d = decide(0.1, 0.5, 0.2)
    assert d.action != "Active"


# ---------------------------------------------------------------------------
# 7. Cfinal identity
# ---------------------------------------------------------------------------


def test_cfinal_identity_on_random_vectors():
    rng = np.random.default_rng(7)
    for _ in range(20):
        vmeta = rng.standard_normal(D)
        vwm = rng.standard_normal(D)
        nmin, nmax = 0.5, 4.0
        C, alpha, Cfinal = confidence_consistency(vmeta, vwm, nmin=nmin, nmax=nmax)
        expected_Cfinal = gated_confidence(C, alpha)
        assert Cfinal == pytest.approx(float(C) * (0.3 + 0.7 * float(alpha)), abs=1e-12)
        assert Cfinal == pytest.approx(expected_Cfinal, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. Action thresholds
# ---------------------------------------------------------------------------


def test_action_thresholds_exact():
    assert THETA == (0.6, 0.4, 0.2)
    assert action_from_cfinal(0.70) == "Active"
    assert action_from_cfinal(0.50) == "Supp"
    assert action_from_cfinal(0.30) == "Silent"
    assert action_from_cfinal(0.10) == "Opt-Out"
    assert action_from_cfinal(0.6) == "Active"
    assert action_from_cfinal(0.4) == "Supp"
    assert action_from_cfinal(0.2) == "Silent"
    assert action_from_cfinal(0.0) == "Opt-Out"


# ---------------------------------------------------------------------------
# 9. M / R extraction
# ---------------------------------------------------------------------------


def test_m_max_cosine_and_r_fallback():
    eq = np.array([1.0, 0.0, 0.0])
    # no relevant memories (all cos <= 0.3)
    ems_none = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    M, R, extra = extract_signals(eq, ems_none)
    assert M == pytest.approx(0.0, abs=1e-12)
    assert R == pytest.approx(0.5)
    assert extra["n_relevant"] == 0

    # exactly one relevant memory
    ems_one = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M, R, extra = extract_signals(eq, ems_one)
    assert M == pytest.approx(1.0, abs=1e-12)
    assert R == pytest.approx(0.5)
    assert extra["n_relevant"] == 1


def test_r_pairwise_mean_when_two_relevant_phi_zero():
    eq = np.array([1.0, 0.0])
    # both memories highly aligned with the query; pairwise cosine is 1
    ems = np.array([[1.0, 0.0], [1.0, 0.0]])
    M, R, extra = extract_signals(eq, ems, texts=["same", "same"])
    assert M == pytest.approx(1.0, abs=1e-12)
    assert extra["n_relevant"] == 2
    assert extra["phi"] == pytest.approx(0.0)
    assert R == pytest.approx(1.0, abs=1e-12)

    # two relevant memories at 90 degrees to each other, both cos>0.3 with query
    eq = np.array([1.0, 1.0])
    ems = np.array([[1.0, 0.0], [0.0, 1.0]])
    M, R, extra = extract_signals(eq, ems)
    # cos(eq, e1) = 1/sqrt(2) ≈ 0.707 > 0.3
    assert extra["n_relevant"] == 2
    assert extra["phi"] == pytest.approx(0.0)
    # pairwise cosine of orthogonal unit vectors is 0
    assert R == pytest.approx(0.0, abs=1e-12)
    assert M == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-12)


# ---------------------------------------------------------------------------
# 10. Lexical conflict φ
# ---------------------------------------------------------------------------


def test_lexical_conflict_phi():
    assert pair_conflict("yes", "no")
    assert pair_conflict("the answer is yes", "the answer is no")
    assert pair_conflict("true", "false")
    assert pair_conflict("safe option", "dangerous option")
    assert pair_conflict("increase", "decrease")
    assert not pair_conflict("hello world", "hello world")
    assert not pair_conflict("the sky is blue", "the sky is blue")

    phi_clash = conflict_rate(["yes it is", "no it is not"])
    phi_ant = conflict_rate(["true statement", "false statement"])
    phi_same = conflict_rate(["identical text", "identical text"])
    assert phi_clash == pytest.approx(1.0)
    assert phi_ant == pytest.approx(1.0)
    assert phi_same == pytest.approx(0.0)

    eq = np.array([1.0, 0.0])
    ems = np.array([[1.0, 0.0], [0.9, 0.1]])
    _, R_clash, extra_clash = extract_signals(eq, ems, texts=["yes", "no"])
    _, R_same, extra_same = extract_signals(eq, ems, texts=["yes", "yes"])
    assert extra_clash["phi"] > extra_same["phi"]
    assert extra_clash["phi"] == pytest.approx(1.0)
    assert extra_same["phi"] == pytest.approx(0.0)
    assert R_clash < R_same


# ---------------------------------------------------------------------------
# 11. α in [-1, 1]
# ---------------------------------------------------------------------------


def test_alpha_in_unit_interval():
    rng = np.random.default_rng(11)
    for _ in range(30):
        vmeta = rng.standard_normal(D)
        vwm = rng.standard_normal(D)
        _, alpha, _ = confidence_consistency(vmeta, vwm, nmin=0.0, nmax=8.0)
        assert -1.0 - 1e-9 <= float(alpha) <= 1.0 + 1e-9

    for M, R, A in [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.5, 0.5, 0.5), (0.9, 0.2, 0.85)]:
        d = decide(M, R, A)
        assert -1.0 <= d.alpha <= 1.0


# ---------------------------------------------------------------------------
# 12. Determinism
# ---------------------------------------------------------------------------


def test_decide_deterministic_seed_42():
    a = decide(0.7, 0.6, 0.3, qr_seed=42)
    b = decide(0.7, 0.6, 0.3, qr_seed=42)
    assert a.action == b.action
    assert a.C == pytest.approx(b.C, abs=1e-12)
    assert a.alpha == pytest.approx(b.alpha, abs=1e-12)
    assert a.Cfinal == pytest.approx(b.Cfinal, abs=1e-12)
    assert a.gA == pytest.approx(b.gA, abs=1e-12)


def test_decide_from_memories_roundtrip():
    eq = np.array([1.0, 0.0, 0.0])
    ems = np.array([[0.95, 0.05, 0.0], [0.9, 0.1, 0.0]])
    d = decide_from_memories(eq, ems, A=0.2, texts=["fact one", "fact one"])
    assert d.action in ("Active", "Supp", "Silent", "Opt-Out")
    assert 0.0 <= d.Cfinal <= 1.0


def test_default_norm_bounds_documented():
    assert NMIN < NMAX
    # documented synthetic calibration band, not an 80-question fit
    assert NMIN == pytest.approx(2.62)
    assert NMAX == pytest.approx(3.32)
