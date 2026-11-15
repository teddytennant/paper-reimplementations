"""Effective-depth diagnostic.

Linear CKA on column-centered matrices:

    CKA(H, H2) = ||H2.T @ H||_F**2 / (||H.T @ H||_F * ||H2.T @ H2||_F)

with CKA = 0 when the denominator is 0.

    rho_hat(k) = 1/(L-k) * sum_{l=1}^{L-k} CKA(H_l, H_{l+k})
    D_eff = L / (1 + 2 * sum_{k=1}^{L-1} (1 - k/L) * rho_hat(k))
    F_L = 2*L/(L+1)
    gap = (F_L - D_eff) / F_L

Indices l and k in the sums are 1-based. rho[k-1] stores rho_hat(k).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def _as_matrix(H) -> jnp.ndarray:
    H = jnp.asarray(H, dtype=jnp.float64)
    if H.ndim == 1:
        H = H.reshape(-1, 1)
    if H.ndim != 2:
        raise ValueError(f"expected an (n, d) matrix, got shape {tuple(H.shape)}")
    return H


def _layer_list(layers) -> list[jnp.ndarray]:
    if isinstance(layers, (list, tuple)):
        mats = [_as_matrix(H) for H in layers]
    else:
        arr = jnp.asarray(layers, dtype=jnp.float64)
        if arr.ndim == 3:
            mats = [_as_matrix(arr[i]) for i in range(arr.shape[0])]
        elif arr.ndim == 2:
            mats = [_as_matrix(arr)]
        else:
            raise ValueError(
                f"expected a list or stack of (n, d) matrices, got shape {tuple(arr.shape)}"
            )
    if len(mats) < 1:
        raise ValueError("layers must be non-empty")
    return mats


def _center_columns(H: jnp.ndarray) -> jnp.ndarray:
    return H - jnp.mean(H, axis=0, keepdims=True)


def cka(H, H2):
    """Linear CKA after column-centering. Returns 0 if the denominator is 0."""
    Hc = _center_columns(_as_matrix(H))
    H2c = _center_columns(_as_matrix(H2))
    cross = H2c.T @ Hc
    gram = Hc.T @ Hc
    gram2 = H2c.T @ H2c
    num = jnp.sum(jnp.square(cross))
    den = jnp.sqrt(jnp.sum(jnp.square(gram))) * jnp.sqrt(jnp.sum(jnp.square(gram2)))
    if float(den) == 0.0:
        return 0.0
    # Identical centered matrices have CKA 1. Float sqrt can miss that by an ulp.
    if Hc.shape == H2c.shape and bool(np.array_equal(np.asarray(Hc), np.asarray(H2c))):
        return 1.0
    return float(num / den)


def orthogonal_update_population_rho(L: int) -> np.ndarray:
    """Population CKA of accumulated orthogonal updates.

    rho[k-1] = 1/(L-k) * sum_{i=1}^{L-k} i/(i+k), with i and k 1-based.
    """
    L = int(L)
    if L < 1:
        raise ValueError("L must be >= 1")
    rho = np.zeros(L - 1, dtype=np.float64)
    for k in range(1, L):
        total = 0.0
        for i in range(1, L - k + 1):
            total += i / (i + k)
        rho[k - 1] = total / (L - k)
    return rho


def deff_from_rho(rho, L: int) -> dict:
    """Effective depth, gap, and F_L from a lag profile rho of length L-1."""
    L = int(L)
    if L < 1:
        raise ValueError("L must be >= 1")
    rho_arr = np.asarray([] if rho is None else rho, dtype=np.float64).reshape(-1)
    if rho_arr.shape[0] != L - 1:
        raise ValueError(f"rho length {rho_arr.shape[0]} != L-1 ({L - 1})")
    ks = np.arange(1, L, dtype=np.float64)
    # (1 - k/L) = (L - k) / L. The integer form is exact when every rho_hat(k) is 1.
    acc = float(np.sum((np.float64(L) - ks) * rho_arr) / np.float64(L))
    depth = float(np.float64(L) / (1.0 + 2.0 * acc))
    f_l = float(2.0 * L / (L + 1))
    gap = float((f_l - depth) / f_l)
    return {
        "deff": depth,
        "rho": rho_arr,
        "gap": gap,
        "f_l": f_l,
    }


def deff(layers) -> dict:
    """Effective depth of a list or stack of (n, d) layer matrices.

    Returns dict with keys deff, rho (length L-1), gap, and f_l.
    """
    mats = _layer_list(layers)
    L = len(mats)
    rho = np.zeros(L - 1, dtype=np.float64)
    for k in range(1, L):
        vals = [cka(mats[l], mats[l + k]) for l in range(L - k)]
        rho[k - 1] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    return deff_from_rho(rho, L)


def deff_on_updates(layers) -> dict:
    """Effective depth of residual updates f_1 = H_1, f_l = H_l - H_{l-1}."""
    mats = _layer_list(layers)
    updates = [mats[0]]
    for prev, cur in zip(mats, mats[1:]):
        updates.append(cur - prev)
    return deff(updates)


def mean_pool(X):
    """Mean over the sequence axis of a token tensor (n, seq, d)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 3:
        raise ValueError(f"mean_pool expects (n, seq, d), got shape {tuple(X.shape)}")
    return np.asarray(jnp.mean(X, axis=1))


def first_non_bos(X):
    """Token at index 1 (first non-BOS position) of a tensor (n, seq, d)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 3:
        raise ValueError(f"first_non_bos expects (n, seq, d), got shape {tuple(X.shape)}")
    if X.shape[1] < 2:
        raise ValueError("first_non_bos needs sequence length >= 2")
    return np.asarray(X[:, 1, :])


def token_normed(X):
    """Divide each token by its L2 norm (floor 1e-12), then mean-pool over seq."""
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 3:
        raise ValueError(f"token_normed expects (n, seq, d), got shape {tuple(X.shape)}")
    norms = jnp.linalg.norm(X, axis=-1, keepdims=True)
    norms = jnp.maximum(norms, 1e-12)
    return np.asarray(jnp.mean(X / norms, axis=1))


def drop_top_pcs(X, k: int):
    """Column-center X (n, d) and remove the top-k principal components.

    The residual is in the column-centered space, so it is orthogonal to each
    removed principal component.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError(f"drop_top_pcs expects (n, d), got shape {tuple(X.shape)}")
    centered = _center_columns(X)
    k = int(k)
    if k <= 0:
        return np.asarray(centered)
    _, _, vh = jnp.linalg.svd(centered, full_matrices=False)
    k_use = min(k, int(vh.shape[0]))
    components = vh[:k_use, :]
    residual = centered - centered @ components.T @ components
    return np.asarray(residual)
