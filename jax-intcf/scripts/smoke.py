"""CPU smoke: Table 1 criteria, one IntCT, one small IntCF."""

from __future__ import annotations

import numpy as np

from jax_intcf import IntCF, IntCT, evaluate_split


def main() -> None:
    t = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    y = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 1.5], dtype=float)
    x1 = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    x2 = np.array([0, 0, 0, 0, 1, 1], dtype=float)
    s1 = evaluate_split(y, t, left_mask=x1 <= 0.5)
    s2 = evaluate_split(y, t, left_mask=x2 <= 0.5)
    print("table1_x1_het", float(s1["het"]))
    print("table1_x1_bias", float(s1["bias"]))
    print("table1_x2_het", float(s2["het"]))
    print("table1_x2_bias", float(s2["bias"]))
    print("table1_intcf_prefers_x1", float(s1["delta"]) > float(s2["delta"]))

    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.integers(0, 2, n).astype(float)
    x1f = rng.integers(0, 2, n).astype(float)
    p = np.where(x0 == 0, 0.25, 0.75)
    tt = rng.binomial(1, p).astype(float)
    yy = 4.0 * x0 + tt * (0.5 + x1f) + rng.normal(0.0, 0.1, n)
    X = np.column_stack([x0, x1f])
    tree = IntCT(max_depth=2, min_leaf_size=8, random_state=0)
    tree.fit(X, yy, tt)
    print("intct_root_feature", tree.root.feature)
    print("intct_root_het_val", tree.root.het_val)
    print("intct_root_bias_val", tree.root.bias_val)
    forest = IntCF(n_trees=5, max_depth=2, min_leaf_size=8, random_state=1)
    forest.fit(X, yy, tt)
    pred = forest.predict(X[:5])
    print("intcf_pred0", float(pred[0]))
    print("intcf_importance", forest.feature_importances_.tolist())


if __name__ == "__main__":
    main()
