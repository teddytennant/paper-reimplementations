"""Subspace metrics and a small synthetic stream generator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray


def principal_angles(Gamma1: Array, Gamma2: Array) -> Array:
    """Principal angles from the SVD of Gamma1^T Gamma2.

    Singular values are clipped to [-1, 1] before arccos.
    """
    G1 = jnp.asarray(Gamma1)
    G2 = jnp.asarray(Gamma2)
    s = jnp.linalg.svd(G1.T @ G2, compute_uv=False)
    s = jnp.clip(s, -1.0, 1.0)
    return jnp.arccos(s)


def grassmann_distance(Gamma1: Array, Gamma2: Array) -> Array:
    """Grassmann distance sqrt(sum_i theta_i^2)."""
    theta = principal_angles(Gamma1, Gamma2)
    return jnp.sqrt(jnp.sum(theta * theta))


def variance_weighted_grassmann(
    Gamma1: Array,
    Gamma2: Array,
    eigenvalues: Array,
) -> Array:
    """Variance-weighted Grassmann distance.

    The paper reports this metric but does not write a formula. This kernel
    uses sqrt(sum_i lambda_i theta_i^2 / sum_i lambda_i), with principal
    angles ordered as in the SVD of Gamma1^T Gamma2 (largest cosine first)
    and ``eigenvalues`` expected in matching decreasing order.
    """
    theta = principal_angles(Gamma1, Gamma2)
    lam = jnp.asarray(eigenvalues)
    lam = lam[: theta.shape[0]]
    denom = jnp.maximum(jnp.sum(lam), 1e-12)
    return jnp.sqrt(jnp.sum(lam * theta * theta) / denom)


def cosine_similarity_matched(Gamma1: Array, Gamma2: Array) -> Array:
    """Absolute cosine similarity of greedily matched columns.

    For each column of Gamma1, pick the still-unmatched column of Gamma2
    with largest |dot product|. Returns a length-d vector of |cosines|
    in Gamma1 column order.
    """
    G1 = np.asarray(Gamma1)
    G2 = np.asarray(Gamma2)
    d = G1.shape[1]
    C = np.abs(G1.T @ G2)
    used = np.zeros(d, dtype=bool)
    out = np.empty(d, dtype=float)
    for i in range(d):
        scores = np.where(used, -np.inf, C[i])
        j = int(np.argmax(scores))
        used[j] = True
        n1 = np.linalg.norm(G1[:, i])
        n2 = np.linalg.norm(G2[:, j])
        denom = max(n1 * n2, 1e-12)
        out[i] = abs(float(np.dot(G1[:, i], G2[:, j]) / denom))
    return jnp.asarray(out)


def generate_synthetic_stream(
    n: int,
    D: int,
    d: int,
    eigenvalues: Array | NDArray[np.floating] | list[float],
    noise_std: float = 0.1,
    seed: int = 0,
    mean: Array | NDArray[np.floating] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples from N(mu, Q Delta Q^T + noise_std^2 I) in R^D, paper Section V.

    Returns (X, Q, mu) with X of shape (n, D) and Q of shape (D, d)
    with orthonormal columns.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(D, d))
    Q, _ = np.linalg.qr(raw)
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.shape != (d,):
        raise ValueError(f"eigenvalues must have shape ({d},), got {lam.shape}")
    cov = Q @ np.diag(lam) @ Q.T + (noise_std**2) * np.eye(D)
    mu_np = np.zeros(D) if mean is None else np.asarray(mean, dtype=float)
    X = rng.multivariate_normal(mu_np, cov, size=n)
    return X, Q, mu_np
