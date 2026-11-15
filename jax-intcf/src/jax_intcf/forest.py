"""Interpretable causal forest (IntCF), algorithm 3.3.2 and 3.3.3."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from jax_intcf.criteria import scale_importances
from jax_intcf.tree import IntCT


class IntCF:
    """Forest of honest IntCT trees with independent train/val splits."""

    def __init__(
        self,
        n_trees: int = 10,
        max_depth: int = 3,
        min_leaf_size: int = 5,
        min_treated: int = 1,
        val_fraction: float = 0.5,
        max_features: int | str | None = "sqrt",
        random_state: int | None = 0,
        importance_criterion: str = "het",
    ) -> None:
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_leaf_size = int(min_leaf_size)
        self.min_treated = int(min_treated)
        self.val_fraction = float(val_fraction)
        self.max_features = max_features
        self.random_state = random_state
        self.importance_criterion = importance_criterion
        self.trees: list[IntCT] = []
        self.n_features_: int | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        train_index: Sequence[int] | None = None,
        val_index: Sequence[int] | None = None,
    ) -> IntCF:
        X = np.asarray(X, dtype=np.float64)
        self.n_features_ = int(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        raw = np.zeros(self.n_features_, dtype=np.float64)
        for _ in range(self.n_trees):
            seed = int(rng.integers(0, 2**31 - 1))
            tree = IntCT(
                max_depth=self.max_depth,
                min_leaf_size=self.min_leaf_size,
                min_treated=self.min_treated,
                val_fraction=self.val_fraction,
                max_features=self.max_features,
                random_state=seed,
            )
            if train_index is None or val_index is None:
                tree.fit(X, y, t)
            else:
                tree.fit(X, y, t, train_index=train_index, val_index=val_index)
            tree.accumulate_importance(raw, criterion=self.importance_criterion)
            self.trees.append(tree)
        self.feature_importances_ = scale_importances(raw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("IntCF must be fit before predict")
        preds = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        return preds.mean(axis=0)
