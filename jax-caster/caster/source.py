"""The source state CASTER keeps at deployment time.

Section 3 stores, in the subspace U_k0 of eq. (1): the projected per-class mean
mu_c, covariance Sigma_c and prior pi_c, plus the global mean m_s and
covariance C_s that eq. (3) needs and the pooled within-class covariance
Sigma_s that eqs. (8) and (9) measure in. Nothing here scales with the number
of source examples, which is the storage claim in Sec. 4.2.

Statistics are computed once in the largest subspace k0 and sliced to the
first k coordinates at test time, as Sec. 3 describes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import CasterConfig
from .linalg import mahalanobis_sq, symmetrize
from .subspace import lda_directions, scatter_matrices, subspace_dim


class SourceState(NamedTuple):
    """Everything CASTER retains from labeled source data."""

    basis: jnp.ndarray        # U_k0, (d, k0)
    means: jnp.ndarray        # mu_c, (C, k0)
    covs: jnp.ndarray         # Sigma_c, (C, k0, k0)
    pooled_cov: jnp.ndarray   # Sigma_s, (k0, k0)
    global_mean: jnp.ndarray  # m_s, (k0,)
    global_cov: jnp.ndarray   # C_s, (k0, k0)
    priors: jnp.ndarray       # pi_c, (C,)
    counts: jnp.ndarray       # n_c, (C,)

    @property
    def num_classes(self) -> int:
        return self.means.shape[0]

    @property
    def k0(self) -> int:
        return self.means.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.basis.shape[0]

    def restrict(self, k: int) -> "SourceState":
        """Slice every statistic to the first k subspace coordinates, eq. (2)."""
        return SourceState(
            basis=self.basis[:, :k],
            means=self.means[:, :k],
            covs=self.covs[:, :k, :k],
            pooled_cov=self.pooled_cov[:k, :k],
            global_mean=self.global_mean[:k],
            global_cov=self.global_cov[:k, :k],
            priors=self.priors,
            counts=self.counts,
        )


def _class_scatter(centered, labels, num_classes, budget=8_000_000):
    """Sum of outer products of `centered` within each class.

    A segment_sum over per-row outer products is O(n k^2), against the
    O(n C k^2) an einsum against the one-hot costs, but it materializes an
    (n, k, k) intermediate. `budget` caps that at a few tens of megabytes by
    walking the rows in blocks.
    """
    k = centered.shape[1]
    rows = max(1, budget // max(k * k, 1))
    total = jnp.zeros((num_classes, k, k), dtype=centered.dtype)
    for start in range(0, centered.shape[0], rows):
        block = centered[start:start + rows]
        outer = block[:, :, None] * block[:, None, :]
        total = total + jax.ops.segment_sum(
            outer, labels[start:start + rows], num_segments=num_classes
        )
    return total


def fit_source(features, labels, num_classes, config: CasterConfig = CasterConfig()):
    """Fit the source state from labeled features.

    `features` is (n, d) and `labels` is (n,) with values in [0, C). The input
    dtype is preserved; pass float64 arrays (with jax_enable_x64) if you care
    about the last few digits of the Mahalanobis quantities.
    """
    h = jnp.asarray(features)
    y = jnp.asarray(labels).astype(jnp.int32)
    n, d = h.shape
    k0 = subspace_dim(num_classes, config.k_max, d)

    s_b, s_w, _, counts, _ = scatter_matrices(h, y, num_classes)
    basis = lda_directions(s_b, s_w, k0, config.within_ridge)

    z = h @ basis
    onehot = (y[:, None] == jnp.arange(num_classes)[None, :]).astype(z.dtype)
    safe = jnp.maximum(counts, 1.0)
    means = (onehot.T @ z) / safe[:, None]

    centered = z - means[y]
    # Per-class covariance with the usual n_c - 1 denominator.
    per_class = _class_scatter(centered, y, num_classes)
    per_class = per_class / jnp.maximum(counts - 1.0, 1.0)[:, None, None]

    # Pooled within-class covariance, denominator n - C, as in the standard
    # LDA estimator. This is the Sigma_s of eqs. (8) and (9).
    pooled = centered.T @ centered / jnp.maximum(n - num_classes, 1)

    global_mean = z.mean(axis=0)
    zc = z - global_mean
    global_cov = zc.T @ zc / max(n - 1, 1)

    return SourceState(
        basis=basis,
        means=means,
        covs=symmetrize(per_class),
        pooled_cov=symmetrize(pooled),
        global_mean=global_mean,
        global_cov=symmetrize(global_cov),
        priors=counts / jnp.sum(counts),
        counts=counts,
    )


def min_class_separation(means, pooled_cov, eps=0.0):
    """gamma_s of eq. (8): the smallest source-class gap in the Sigma_s metric.

    Classes with no samples must be dropped by the caller; an all-zero mean row
    would otherwise sit at the origin and drag gamma_s down.
    """
    means = jnp.asarray(means)
    c = means.shape[0]
    diffs = means[:, None, :] - means[None, :, :]
    dists = jnp.sqrt(jnp.maximum(mahalanobis_sq(pooled_cov, diffs.reshape(c * c, -1)), 0.0))
    dists = dists.reshape(c, c)
    off_diagonal = dists + jnp.eye(c, dtype=dists.dtype) * jnp.max(dists) * 2.0
    return jnp.min(off_diagonal) + eps
