"""DKRGD: local KRGD on disjoint shards, then weighted aggregation (5).

    f̄_{t,D} = Σ_j (|D_j|/|D|) f_{t, D_j}.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax.numpy as jnp

from dkrgd.krgd import KRGDModel, fit_krgd, predict_krgd
from dkrgd.losses import LossName


class LocalModel(NamedTuple):
    model: KRGDModel
    weight: float  # |D_j| / |D|


def partition(X, y, m: int) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Split (X, y) into m contiguous disjoint blocks covering D."""
    X = jnp.asarray(X)
    y = jnp.asarray(y).reshape(-1)
    n = int(y.shape[0])
    if m < 1 or m > n:
        raise ValueError(f"need 1 ≤ m ≤ n, got m={m}, n={n}")
    sizes = [n // m + (1 if i < n % m else 0) for i in range(m)]
    out = []
    start = 0
    for size in sizes:
        out.append((X[start : start + size], y[start : start + size]))
        start += size
    return out


def fit_dkrgd(
    X,
    y,
    *,
    m: int,
    n_steps: int,
    eta: float,
    sigma: float,
    loss: LossName = "welsch",
    lengthscale: float = 1.0,
) -> list[LocalModel]:
    """Train one local KRGD estimator per block of a partition into m machines."""
    X = jnp.asarray(X)
    y = jnp.asarray(y).reshape(-1)
    n = int(y.shape[0])
    shards = partition(X, y, m)
    locals_ = []
    for Xj, yj in shards:
        model = fit_krgd(
            Xj,
            yj,
            n_steps=n_steps,
            eta=eta,
            sigma=sigma,
            loss=loss,
            lengthscale=lengthscale,
        )
        locals_.append(LocalModel(model=model, weight=model.n / n))
    return locals_


def predict_dkrgd(locals_: Sequence[LocalModel], X_new):
    """Weighted average of local RKHS functions, eq. (5)."""
    X_new = jnp.asarray(X_new)
    acc = jnp.zeros((X_new.shape[0],), dtype=X_new.dtype)
    for loc in locals_:
        acc = acc + loc.weight * predict_krgd(loc.model, X_new)
    return acc
