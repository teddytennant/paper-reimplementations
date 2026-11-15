"""QR-derived complementary projectors and leakage-aware weights W_k (Eq. 1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from mdl.constants import D, EPS_C, GAINS, QR_SEED, SUBSPACE_DIMS

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Subspaces:
    """Frozen QR split of R^D into complementary (wm, r, a) subspaces.

    Rrand is 16x16, drawn from numpy.random.default_rng(seed).standard_normal.
    Q comes from numpy.linalg.qr(Rrand). Columns of Q are split 5/5/6.
    """

    Q: jnp.ndarray
    P_wm: jnp.ndarray
    P_r: jnp.ndarray
    P_a: jnp.ndarray
    Pi_wm: jnp.ndarray
    Pi_r: jnp.ndarray
    Pi_a: jnp.ndarray
    seed: int


def _qr_basis_numpy(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    rrand = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(rrand)
    return np.asarray(Q, dtype=np.float64)


@lru_cache(maxsize=8)
def build_subspaces(seed: int = QR_SEED) -> Subspaces:
    Q_np = _qr_basis_numpy(seed)
    dwm, dr, da = SUBSPACE_DIMS
    assert dwm + dr + da == D
    P_wm = Q_np[:, :dwm]
    P_r = Q_np[:, dwm : dwm + dr]
    P_a = Q_np[:, dwm + dr :]
    Pi_wm = P_wm @ P_wm.T
    Pi_r = P_r @ P_r.T
    Pi_a = P_a @ P_a.T
    return Subspaces(
        Q=jnp.asarray(Q_np),
        P_wm=jnp.asarray(P_wm),
        P_r=jnp.asarray(P_r),
        P_a=jnp.asarray(P_a),
        Pi_wm=jnp.asarray(Pi_wm),
        Pi_r=jnp.asarray(Pi_r),
        Pi_a=jnp.asarray(Pi_a),
        seed=int(seed),
    )


def subspace_weights(sub: Subspaces, eps_c: float = EPS_C, gains=GAINS) -> dict:
    """W_k = s_k Π_k + ε_c * sum_{j≠k} Π_j  (Eq. 1)."""
    s_wm, s_r, s_a = gains
    Pi_wm, Pi_r, Pi_a = sub.Pi_wm, sub.Pi_r, sub.Pi_a
    W_wm = s_wm * Pi_wm + eps_c * (Pi_r + Pi_a)
    W_r = s_r * Pi_r + eps_c * (Pi_wm + Pi_a)
    W_a = s_a * Pi_a + eps_c * (Pi_wm + Pi_r)
    return {"wm": W_wm, "r": W_r, "a": W_a}
