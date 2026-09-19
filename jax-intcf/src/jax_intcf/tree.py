"""Honest interpretable causal tree (IntCT), algorithm 3.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jax_intcf.criteria import (
    bias_criterion,
    bias_validation,
    difference_in_means,
    het_criterion,
    intcf_delta,
)


def _as_data(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    t = (np.asarray(t).reshape(-1) != 0).astype(np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if X.shape[0] != y.shape[0] or y.shape[0] != t.shape[0]:
        raise ValueError("X, y, t must share the first dimension")
    return X, y, t


def _finite_tau(y: np.ndarray, t: np.ndarray, fallback: float) -> float:
    if y.size == 0:
        return float(fallback)
    tau = float(difference_in_means(y, t))
    if not np.isfinite(tau):
        return float(fallback)
    return tau


def _max_features_count(d: int, max_features: int | str | None) -> int:
    if max_features is None:
        return d
    if max_features == "sqrt":
        return max(1, int(np.sqrt(d)))
    if max_features == "log2":
        return max(1, int(np.log2(d)))
    m = int(max_features)
    return min(d, max(1, m))


def _choose_features(
    rng: np.random.Generator,
    d: int,
    max_features: int | str | None,
) -> np.ndarray:
    m = _max_features_count(d, max_features)
    if m >= d:
        return np.arange(d, dtype=int)
    return rng.choice(d, size=m, replace=False)


@dataclass
class Node:
    is_leaf: bool
    n_train: int
    tau_train: float
    prediction: float = 0.0
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
    het: float | None = None
    bias: float | None = None
    delta: float | None = None
    het_val: float | None = None
    bias_val: float | None = None
    tau_forward: float | None = None
    tau_backward: float | None = None
    n_val: int = 0


@dataclass
class _Candidate:
    feature: int
    threshold: float
    het: float
    bias: float
    delta: float
    n_L: int
    n_R: int
    tau_L: float
    tau_R: float


def _best_split_on_feature(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    N: float,
    min_leaf_size: int,
    min_treated: int,
    tau_parent: float,
    feature: int,
) -> _Candidate | None:
    n = int(y.shape[0])
    if n < 2 * min_leaf_size:
        return None
    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    y_s = y[order]
    t_s = t[order]
    n1 = np.cumsum(t_s)
    n0 = np.cumsum(1.0 - t_s)
    s1 = np.cumsum(y_s * t_s)
    s0 = np.cumsum(y_s * (1.0 - t_s))
    k = np.arange(1, n)
    n_L = k.astype(np.float64)
    n_R = float(n) - n_L
    n1_L = n1[:-1]
    n0_L = n0[:-1]
    n1_R = n1[-1] - n1_L
    n0_R = n0[-1] - n0_L
    tau_L = s1[:-1] / np.maximum(n1_L, 1.0) - s0[:-1] / np.maximum(n0_L, 1.0)
    tau_R = (s1[-1] - s1[:-1]) / np.maximum(n1_R, 1.0) - (
        s0[-1] - s0[:-1]
    ) / np.maximum(n0_R, 1.0)
    distinct = x_s[1:] > x_s[:-1]
    valid = (
        distinct
        & (n_L >= min_leaf_size)
        & (n_R >= min_leaf_size)
        & (n1_L >= min_treated)
        & (n0_L >= min_treated)
        & (n1_R >= min_treated)
        & (n0_R >= min_treated)
    )
    if not np.any(valid):
        return None
    het = np.asarray(het_criterion(n_L, n_R, N, tau_L, tau_R), dtype=np.float64)
    bias = np.asarray(
        bias_criterion(n_L, n_R, N, tau_parent, tau_L, tau_R),
        dtype=np.float64,
    )
    delta = np.asarray(intcf_delta(het, bias), dtype=np.float64)
    delta = np.where(valid, delta, -np.inf)
    idx = int(np.argmax(delta))
    if not np.isfinite(delta[idx]):
        return None
    return _Candidate(
        feature=feature,
        threshold=0.5 * float(x_s[idx] + x_s[idx + 1]),
        het=float(het[idx]),
        bias=float(bias[idx]),
        delta=float(delta[idx]),
        n_L=int(n_L[idx]),
        n_R=int(n_R[idx]),
        tau_L=float(tau_L[idx]),
        tau_R=float(tau_R[idx]),
    )


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    N: float,
    min_leaf_size: int,
    min_treated: int,
    tau_parent: float,
    feature_indices: np.ndarray,
) -> _Candidate | None:
    best: _Candidate | None = None
    for j in feature_indices:
        cand = _best_split_on_feature(
            X[:, int(j)],
            y,
            t,
            N,
            min_leaf_size,
            min_treated,
            tau_parent,
            int(j),
        )
        if cand is None:
            continue
        if best is None or cand.delta > best.delta:
            best = cand
    return best


class IntCT:
    """Interpretable causal tree with honest train/validation split."""

    def __init__(
        self,
        max_depth: int = 3,
        min_leaf_size: int = 5,
        min_treated: int = 1,
        val_fraction: float = 0.5,
        max_features: int | str | None = None,
        random_state: int | None = 0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_leaf_size = int(min_leaf_size)
        self.min_treated = int(min_treated)
        self.val_fraction = float(val_fraction)
        self.max_features = max_features
        self.random_state = random_state
        self.root: Node | None = None
        self.n_features_: int | None = None
        self.train_index_: np.ndarray | None = None
        self.val_index_: np.ndarray | None = None
        self.val_predictions_: np.ndarray | None = None
        self.N_train_: int | None = None
        self.N_val_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        train_index: Sequence[int] | None = None,
        val_index: Sequence[int] | None = None,
    ) -> IntCT:
        X, y, t = _as_data(X, y, t)
        n, d = X.shape
        self.n_features_ = d
        rng = np.random.default_rng(self.random_state)
        if train_index is None or val_index is None:
            perm = rng.permutation(n)
            n_val = int(round(n * self.val_fraction))
            n_val = min(max(n_val, 1 if n > 1 else 0), n - 1 if n > 1 else n)
            val_index = perm[:n_val]
            train_index = perm[n_val:]
        train_index = np.asarray(train_index, dtype=int)
        val_index = np.asarray(val_index, dtype=int)
        self.train_index_ = train_index
        self.val_index_ = val_index
        Xj, yj, tj = X[train_index], y[train_index], t[train_index]
        N = float(yj.shape[0])
        self.N_train_ = int(N)
        self.root = self._grow(Xj, yj, tj, depth=0, N=N, rng=rng, fallback=0.0)
        Xi, yi, ti = X[val_index], y[val_index], t[val_index]
        self.N_val_ = int(yi.shape[0])
        val_pred = np.zeros(self.N_val_, dtype=np.float64)
        self._fill_honest(
            self.root,
            Xi,
            yi,
            ti,
            np.arange(self.N_val_),
            float(self.N_val_),
            val_pred,
            fallback=float(self.root.tau_train),
        )
        self.val_predictions_ = val_pred
        return self

    def _grow(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        depth: int,
        N: float,
        rng: np.random.Generator,
        fallback: float,
    ) -> Node:
        n = int(y.shape[0])
        tau_train = _finite_tau(y, t, fallback)
        leaf = Node(
            is_leaf=True,
            n_train=n,
            tau_train=tau_train,
            prediction=tau_train,
        )
        if (
            depth >= self.max_depth
            or n < 2 * self.min_leaf_size
            or not np.isfinite(float(difference_in_means(y, t)))
        ):
            return leaf
        features = _choose_features(rng, X.shape[1], self.max_features)
        cand = _best_split(
            X,
            y,
            t,
            N,
            self.min_leaf_size,
            self.min_treated,
            tau_train,
            features,
        )
        if cand is None or cand.delta < 0.0:
            return leaf
        left_mask = X[:, cand.feature] <= cand.threshold
        if not np.any(left_mask) or np.all(left_mask):
            return leaf
        node = Node(
            is_leaf=False,
            n_train=n,
            tau_train=tau_train,
            prediction=tau_train,
            feature=cand.feature,
            threshold=cand.threshold,
            het=cand.het,
            bias=cand.bias,
            delta=cand.delta,
        )
        node.left = self._grow(
            X[left_mask],
            y[left_mask],
            t[left_mask],
            depth + 1,
            N,
            rng,
            fallback=tau_train,
        )
        node.right = self._grow(
            X[~left_mask],
            y[~left_mask],
            t[~left_mask],
            depth + 1,
            N,
            rng,
            fallback=tau_train,
        )
        return node

    def _fill_honest(
        self,
        node: Node,
        X: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        indices: np.ndarray,
        N_val: float,
        val_pred: np.ndarray,
        fallback: float,
    ) -> None:
        node.n_val = int(indices.shape[0])
        if node.n_val == 0:
            node.tau_forward = fallback
            node.tau_backward = fallback
            node.prediction = fallback
            node.het_val = 0.0
            node.bias_val = 0.0
            return
        tau_f = _finite_tau(y, t, fallback)
        node.tau_forward = tau_f
        if node.is_leaf:
            node.prediction = tau_f
            node.tau_backward = tau_f
            val_pred[indices] = tau_f
            return
        assert node.left is not None and node.right is not None
        left_mask = X[:, int(node.feature)] <= float(node.threshold)
        self._fill_honest(
            node.left,
            X[left_mask],
            y[left_mask],
            t[left_mask],
            indices[left_mask],
            N_val,
            val_pred,
            fallback=tau_f,
        )
        self._fill_honest(
            node.right,
            X[~left_mask],
            y[~left_mask],
            t[~left_mask],
            indices[~left_mask],
            N_val,
            val_pred,
            fallback=tau_f,
        )
        n_L = node.left.n_val
        n_R = node.right.n_val
        if n_L + n_R == 0:
            node.tau_backward = tau_f
        else:
            node.tau_backward = (
                n_L * float(node.left.tau_backward)
                + n_R * float(node.right.tau_backward)
            ) / (n_L + n_R)
        if n_L > 0 and n_R > 0:
            node.het_val = float(
                het_criterion(
                    n_L,
                    n_R,
                    N_val,
                    node.left.tau_backward,
                    node.right.tau_backward,
                )
            )
            node.bias_val = float(
                bias_validation(
                    n_L,
                    n_R,
                    N_val,
                    node.tau_forward,
                    node.tau_backward,
                    node.left.tau_forward,
                    node.left.tau_backward,
                    node.right.tau_forward,
                    node.right.tau_backward,
                )
            )
        else:
            node.het_val = 0.0
            node.bias_val = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("IntCT must be fit before predict")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            node = self.root
            while not node.is_leaf:
                if X[i, int(node.feature)] <= float(node.threshold):
                    node = node.left  # type: ignore[assignment]
                else:
                    node = node.right  # type: ignore[assignment]
            out[i] = node.prediction
        return out

    def accumulate_importance(self, scores: np.ndarray, criterion: str = "het") -> None:
        if self.root is None:
            return
        _accumulate(self.root, scores, criterion)


def _accumulate(node: Node, scores: np.ndarray, criterion: str) -> None:
    if node.is_leaf or node.feature is None:
        return
    if criterion == "het":
        scores[node.feature] += float(node.het_val or 0.0)
    elif criterion == "bias":
        scores[node.feature] += float(node.bias_val or 0.0)
    elif criterion == "intcf":
        scores[node.feature] += float(max(node.het_val or 0.0, node.bias_val or 0.0))
    else:
        scores[node.feature] += float(node.het_val or 0.0) + float(node.bias_val or 0.0)
    if node.left is not None:
        _accumulate(node.left, scores, criterion)
    if node.right is not None:
        _accumulate(node.right, scores, criterion)
