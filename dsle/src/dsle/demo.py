"""Synthetic path-graph demo for DSLE."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from dsle.estimate import (
    dsle,
    laplacian_from_resistance,
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
    evals, evecs = jnp.linalg.eigh(0.5 * (L + L.T))
    inv = jnp.where(evals > 1e-8, 1.0 / evals, 0.0)
    S = (evecs * inv) @ evecs.T
    p = L.shape[0]
    J = jnp.eye(p, dtype=jnp.float64) - jnp.ones((p, p), dtype=jnp.float64) / p
    return J @ S @ J


def sample_igmrf(L: jnp.ndarray, n: int, seed: int) -> jnp.ndarray:
    """Draw n samples from N(0, L^+), zeroing the constant mode."""
    evals, evecs = jnp.linalg.eigh(0.5 * (L + L.T))
    inv_sqrt = jnp.where(evals > 1e-8, jnp.sqrt(1.0 / evals), 0.0)
    z = jax.random.normal(jax.random.PRNGKey(seed), (n, L.shape[0]), dtype=jnp.float64)
    return (z * inv_sqrt) @ evecs.T


def _fscore(estimated: set[tuple[int, int]], truth: set[tuple[int, int]]) -> float:
    tp = len(estimated & truth)
    fp = len(estimated - truth)
    fn = len(truth - estimated)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def _edges(L: jnp.ndarray, tol: float = 1e-8) -> set[tuple[int, int]]:
    p = L.shape[0]
    scale = float(jnp.max(jnp.abs(L)))
    cutoff = tol * max(scale, 1.0)
    found: set[tuple[int, int]] = set()
    for i in range(p):
        for j in range(i + 1, p):
            if abs(float(L[i, j])) > cutoff:
                found.add((i, j))
    return found


def main() -> None:
    p = 12
    L = path_laplacian(p)
    S = laplacian_pinv(L)
    R = resistance_from_covariance(S)
    recovered = laplacian_from_resistance(R)
    rel = float(jnp.linalg.norm(recovered - L) / jnp.linalg.norm(L))
    print(f"path p={p} inverse relative error {rel:.3e}")

    p = 24
    n = 12
    gamma = 1.5
    M = 8
    seed = 0
    L = path_laplacian(p)
    X = sample_igmrf(L, n, seed)
    S = sample_covariance(X)
    L_bar = dsle(X, gamma)
    laps = sparsify(L_bar, S, M)
    truth = {(i, i + 1) for i in range(p - 1)}
    counts = [len(_edges(Lm)) for Lm in laps]
    target = p - 1
    m = min(range(M), key=lambda k: (abs(counts[k] - target), k))
    fs = _fscore(_edges(laps[m]), truth)
    tr = float(jnp.trace(laps[m] @ S))
    print(f"path p={p} n={n} gamma={gamma} M={M} seed={seed}")
    print(f"F-score {fs:.3f} at m={m + 1} with {counts[m]} edges")
    print(f"trace(L_m S) {tr:.6f}")


if __name__ == "__main__":
    main()
