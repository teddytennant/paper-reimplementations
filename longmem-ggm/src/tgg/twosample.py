"""Directional two-sample test.

Estimate a graph from sample 1 and test sample 2 against that support, as in
the two-sample corollary. The corollary cites Chen, Xu, and Wu (2013) as an
example of a consistent precision estimator. This module uses a nodewise
ridge precision, converted to partial correlations and thresholded, as a
stand-in for that estimator. The decision is the direct test on sample 2.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from tgg.core import direct_test, sample_covariance


def nodewise_ridge_precision(X, ridge: float = 1e-2) -> jnp.ndarray:
    """Nodewise ridge precision. Columns are not constrained to a graph."""

    Sigma = np.asarray(sample_covariance(X), dtype=np.float64)
    p = int(Sigma.shape[0])
    omega = np.zeros((p, p), dtype=np.float64)
    for j in range(p):
        if p == 1:
            omega[0, 0] = 1.0 / max(float(Sigma[0, 0]), 1e-8)
            continue
        idx = np.array([i for i in range(p) if i != j], dtype=np.int64)
        gram = Sigma[np.ix_(idx, idx)] + float(ridge) * np.eye(p - 1)
        cross = Sigma[idx, j]
        beta = np.linalg.solve(gram, cross)
        var = max(float(Sigma[j, j] - cross @ beta), 1e-8)
        scale = 1.0 / var
        omega[idx, j] = -scale * beta
        omega[j, j] = scale
    omega = 0.5 * (omega + omega.T)
    return jnp.asarray(omega)


def partial_correlations(omega) -> jnp.ndarray:
    """rho_ij = -Omega_ij / sqrt(Omega_ii Omega_jj)."""

    omega = jnp.asarray(omega, dtype=jnp.float64)
    scale = jnp.sqrt(jnp.maximum(jnp.diag(omega), 1e-12))
    return -omega / (scale[:, None] * scale[None, :])


def thresholded_partial_correlation_graph(X, threshold: float = 0.2, ridge: float = 1e-2):
    """Undirected edges where the absolute partial correlation exceeds threshold.

    Documented stand-in for the Chen, Xu, and Wu (2013) precision estimator
    cited in the two-sample corollary. Not the Sherlock threshold path.
    """

    rho = np.asarray(partial_correlations(nodewise_ridge_precision(X, ridge=ridge)))
    p = rho.shape[0]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if abs(float(rho[i, j])) > threshold:
                edges.append((i, j))
    return edges


def two_sample_test(
    X1,
    X2,
    l=None,
    alpha: float = 0.05,
    threshold: float = 0.2,
    ridge: float = 1e-2,
) -> bool:
    """Test sample 2 against a graph estimated from sample 1.

    Returns the reject decision as a bool.
    """

    edges = thresholded_partial_correlation_graph(X1, threshold=threshold, ridge=ridge)
    result = direct_test(X2, edges, l=l, alpha=alpha)
    return bool(result.reject)
