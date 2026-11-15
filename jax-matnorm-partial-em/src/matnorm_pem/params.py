"""Parameter container and the scale constraint of Section 2.4.

The Kronecker factors of a matrix normal are only identified up to a scalar,
since Sigma_2 (x) Sigma_1 = (Sigma_2 / alpha) (x) (alpha Sigma_1). Section 2.4
pins this down by pulling out a variance scalar and requiring |Sigma_1| = 1 and
|Sigma_2| = 1, so the full covariance is

    Sigma = sigma^2 (Sigma_2 (x) Sigma_1),   Xi = (1 / sigma^2) (Xi_2 (x) Xi_1).
"""

from typing import NamedTuple

import jax.numpy as jnp


class MatNormParams(NamedTuple):
    """Matrix normal parameters under the Section 2.4 constraint.

    M: p x q location matrix.
    S1: p x p row covariance factor, determinant 1.
    S2: q x q column covariance factor, determinant 1.
    sigma2: scalar variance carrying the overall scale.
    """

    M: jnp.ndarray
    S1: jnp.ndarray
    S2: jnp.ndarray
    sigma2: jnp.ndarray

    @property
    def p(self) -> int:
        return self.M.shape[0]

    @property
    def q(self) -> int:
        return self.M.shape[1]


def normalize_scale(M, S1, S2, sigma2=1.0) -> MatNormParams:
    """Move all of the scale into sigma^2 so that |S1| = |S2| = 1.

    Applies the normalizations in (11) and (12) to an arbitrary (S1, S2, sigma^2)
    triple. Use this before comparing fitted factors against the truth: two
    parameter sets that differ only by the Kronecker scalar map to the same
    normalized triple.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    S1 = jnp.asarray(S1, dtype=jnp.float64)
    S2 = jnp.asarray(S2, dtype=jnp.float64)
    p, q = S1.shape[0], S2.shape[0]
    d1 = jnp.linalg.det(S1) ** (1.0 / p)
    d2 = jnp.linalg.det(S2) ** (1.0 / q)
    return MatNormParams(
        M=M,
        S1=S1 / d1,
        S2=S2 / d2,
        sigma2=jnp.asarray(sigma2, dtype=jnp.float64) * d1 * d2,
    )
