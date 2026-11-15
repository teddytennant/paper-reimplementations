"""Equation-level tests for IntCT / IntCF (arXiv:2609.16971)."""

from __future__ import annotations

import numpy as np
import pytest

from jax_intcf import (
    IntCF,
    IntCT,
    bias_criterion,
    difference_in_means,
    evaluate_split,
    het_criterion,
    lemma4_mse_decrease,
    unsigned_from_signed,
)


def _table1():
    """Paper Table 1: X1 confounder, X2 spuriously heterogeneous."""
    t = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], dtype=float)
    x1 = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    x2 = np.array([0, 0, 0, 0, 1, 1], dtype=float)
    X = np.column_stack([x1, x2])
    return X, y, t


def test_eq11_15_16_confounded_split():
    X, y, t = _table1()
    stats = evaluate_split(y, t, left_mask=X[:, 0] <= 0.5)
    assert float(stats["tau_parent"]) == pytest.approx(1.0 / 3.0)
    assert float(stats["tau_L"]) == pytest.approx(0.0)
    assert float(stats["tau_R"]) == pytest.approx(0.0)
    assert float(stats["het"]) == pytest.approx(0.0)
    assert float(stats["bias"]) == pytest.approx(1.0 / 9.0)
    assert float(stats["mse"]) == pytest.approx(float(stats["het"]) + float(stats["bias"]))
    assert float(stats["delta"]) == pytest.approx(1.0 / 9.0)


def test_eq11_15_16_heterogeneous_split():
    # Propensity balanced, tau changes, parent ATE equals weighted children.
    t = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y = np.array([0, 2, 0, 2, 0, 0, 0, 0], dtype=float)
    left = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    stats = evaluate_split(y, t, left_mask=left)
    assert float(stats["tau_L"]) == pytest.approx(2.0)
    assert float(stats["tau_R"]) == pytest.approx(0.0)
    assert float(stats["tau_parent"]) == pytest.approx(1.0)
    assert float(stats["bias"]) == pytest.approx(0.0)
    assert float(stats["het"]) == pytest.approx(1.0)
    assert float(stats["mse"]) == pytest.approx(float(stats["het"]) + float(stats["bias"]))
    assert float(stats["delta"]) == pytest.approx(1.0)


def test_eq16_identity_table1_x2():
    X, y, t = _table1()
    stats = evaluate_split(y, t, left_mask=X[:, 1] <= 0.5)
    assert float(stats["het"]) == pytest.approx(1.0 / 18.0)
    assert float(stats["bias"]) == pytest.approx(0.0)
    assert float(stats["mse"]) == pytest.approx(float(stats["het"]) + float(stats["bias"]))


def test_signed_recover_unsigned_eq18_19():
    X, y, t = _table1()
    for col in (0, 1):
        stats = evaluate_split(y, t, left_mask=X[:, col] <= 0.5)
        het_u = unsigned_from_signed(
            stats["signed_het"], stats["n_L"], stats["n_R"], stats["N"]
        )
        bias_u = unsigned_from_signed(
            stats["signed_bias"], stats["n_L"], stats["n_R"], stats["N"]
        )
        assert float(het_u) == pytest.approx(float(stats["het"]), abs=1e-12)
        assert float(bias_u) == pytest.approx(float(stats["bias"]), abs=1e-12)


def test_confounder_only_bias_exceeds_het():
    X, y, t = _table1()
    stats = evaluate_split(y, t, left_mask=X[:, 0] <= 0.5)
    assert float(stats["bias"]) > float(stats["het"])
    rng = np.random.default_rng(1)
    n = 400
    x0 = rng.integers(0, 2, n).astype(float)
    p = np.where(x0 == 0, 0.2, 0.8)
    tt = rng.binomial(1, p).astype(float)
    yy = 5.0 * x0 + 1.0 * tt + rng.normal(0.0, 0.05, n)
    stats2 = evaluate_split(yy, tt, left_mask=x0 <= 0.5)
    assert float(stats2["bias"]) > float(stats2["het"])


def test_heterogeneity_only_het_exceeds_bias():
    t = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y = np.array([0, 2, 0, 2, 0, 0, 0, 0], dtype=float)
    left = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    stats = evaluate_split(y, t, left_mask=left)
    assert float(stats["het"]) > float(stats["bias"])
    rng = np.random.default_rng(2)
    n = 400
    x1 = rng.integers(0, 2, n).astype(float)
    tt = rng.binomial(1, 0.5, n).astype(float)
    tau = 0.5 + 2.0 * x1
    yy = tt * tau + rng.normal(0.0, 0.05, n)
    stats2 = evaluate_split(yy, tt, left_mask=x1 <= 0.5)
    assert float(stats2["het"]) > float(stats2["bias"])


def _toy_confounder_modifier(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(float)
    x1 = rng.integers(0, 2, n).astype(float)
    p = np.where(x0 == 0, 0.2, 0.8)
    t = rng.binomial(1, p).astype(float)
    mu0 = 5.0 * x0
    tau = 0.5 + 1.5 * x1
    y = mu0 + t * tau + rng.normal(0.0, 0.05, n)
    X = np.column_stack([x0, x1])
    return X, y, t


def test_intcf_prefers_confounder_at_root():
    X, y, t = _toy_confounder_modifier(n=600, seed=0)
    s0 = evaluate_split(y, t, left_mask=X[:, 0] <= 0.5)
    s1 = evaluate_split(y, t, left_mask=X[:, 1] <= 0.5)
    assert float(s0["bias"]) > float(s0["het"])
    assert float(s1["het"]) > float(s1["bias"])
    assert float(s0["delta"]) > float(s1["delta"])
    tree = IntCT(
        max_depth=2,
        min_leaf_size=20,
        min_treated=1,
        val_fraction=0.5,
        max_features=None,
        random_state=0,
    )
    tree.fit(X, y, t)
    assert tree.root is not None
    assert not tree.root.is_leaf
    assert tree.root.feature == 0


def test_honest_leaf_uses_validation_not_training():
    X = np.zeros((10, 1), dtype=float)
    t = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    tree = IntCT(max_depth=0, min_leaf_size=1, random_state=0)
    tree.fit(X, y, t, train_index=np.arange(5), val_index=np.arange(5, 10))
    pred = float(tree.predict(X[:1])[0])
    tau_val = float(difference_in_means(y[5:], t[5:]))
    tau_tr = float(difference_in_means(y[:5], t[:5]))
    assert tau_val == pytest.approx(0.0)
    assert tau_tr == pytest.approx(10.0)
    assert pred == pytest.approx(tau_val)
    assert pred != pytest.approx(tau_tr)


def test_lemma4_identity_on_grown_stump():
    X0, y0, t0 = _table1()
    X = np.vstack([X0] * 8)
    y = np.tile(y0, 8)
    t = np.tile(t0, 8)
    idx = np.arange(len(y))
    tree = IntCT(max_depth=1, min_leaf_size=2, min_treated=1, random_state=0)
    tree.fit(X, y, t, train_index=idx, val_index=idx)
    root = tree.root
    assert root is not None and not root.is_leaf
    Xv = X[tree.val_index_]
    left = Xv[:, int(root.feature)] <= float(root.threshold)
    tau_i = tree.val_predictions_
    N = float(len(tau_i))
    lhs = float(root.het_val) + float(root.bias_val)
    rhs = float(
        lemma4_mse_decrease(
            root.tau_forward,
            root.left.tau_forward,
            root.right.tau_forward,
            tau_i,
            tau_i[left],
            tau_i[~left],
            N,
        )
    )
    assert lhs == pytest.approx(rhs, abs=1e-10)


def test_forest_predict_mean_and_importance_sums_to_one():
    X, y, t = _toy_confounder_modifier(n=240, seed=3)
    forest = IntCF(
        n_trees=5,
        max_depth=2,
        min_leaf_size=10,
        max_features="sqrt",
        random_state=0,
        importance_criterion="het",
    )
    forest.fit(X, y, t)
    Xt = X[:20]
    pred = forest.predict(Xt)
    mean_trees = np.mean([tr.predict(Xt) for tr in forest.trees], axis=0)
    np.testing.assert_allclose(pred, mean_trees, atol=1e-12)
    imp = forest.feature_importances_
    assert imp is not None
    assert imp.shape == (2,)
    assert np.all(imp >= -1e-12)
    assert float(np.maximum(imp, 0.0).sum()) == pytest.approx(1.0, abs=1e-10)


def test_intcf_delta_is_max_het_bias():
    het = het_criterion(3.0, 3.0, 6.0, 0.0, 0.0)
    bias = bias_criterion(3.0, 3.0, 6.0, 1.0 / 3.0, 0.0, 0.0)
    from jax_intcf import intcf_delta

    assert float(intcf_delta(het, bias)) == pytest.approx(max(float(het), float(bias)))
