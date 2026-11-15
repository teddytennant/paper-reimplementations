"""Problem data for robust sparse mean-variance selection (P1)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random


@dataclass(frozen=True)
class RobustSparseMV:
    """Instance of (P1) in excess-return coordinates.

    Attributes
    ----------
    D :
        SPD covariance in ``R^{N x N}``.
    r :
        Excess-return vector ``r := r_hat - r_c 1``.
    r_bar :
        Excess target ``r_bar := target - r_c``. Assumption 1 requires ``r_bar > 0``.
    gamma :
        Ellipsoidal uncertainty radius ``γ > 0``.
    beta :
        ``ℓ0`` penalty ``β > 0``.
    """

    D: Array
    r: Array
    r_bar: float
    gamma: float
    beta: float

    @property
    def n(self) -> int:
        return int(self.D.shape[0])


def synthetic_instance(
    n: int = 6,
    *,
    seed: int = 0,
    gamma: float = 0.05,
    r_bar: float = 0.08,
    beta: float = 0.02,
    margin: float = 0.35,
    dtype=jnp.float64,
) -> RobustSparseMV:
    """SPD covariance ``D = A A^T + 0.1 I`` with excess returns satisfying Assumption 2.

    Each coordinate obeys ``|r_i| / sqrt(D_ii) = gamma + margin``, so every singleton
    (and every nonempty support) is strictly feasible.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if r_bar <= 0:
        raise ValueError("r_bar must be positive (Assumption 1)")
    if margin <= 0:
        raise ValueError("margin must be positive to enforce Assumption 2")

    key = random.PRNGKey(seed)
    k_a, k_s = random.split(key)
    a = random.normal(k_a, (n, n), dtype=dtype)
    d = a @ a.T + 0.1 * jnp.eye(n, dtype=dtype)
    vol = jnp.sqrt(jnp.diag(d))
    signs = jnp.where(random.bernoulli(k_s, 0.85, (n,)), 1.0, -1.0).astype(dtype)
    r = signs * (gamma + margin) * vol
    return RobustSparseMV(D=d, r=r, r_bar=float(r_bar), gamma=float(gamma), beta=float(beta))
