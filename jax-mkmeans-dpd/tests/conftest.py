import jax
import numpy as np
import pytest

# The convergence and stationarity tests make claims about exact arithmetic:
# "Q never goes up", "the gradient is zero". In float32 a sum of several hundred
# partly cancelling terms carries a few parts in 1e6 of noise, which is larger
# than the last real decreases of Q near a fixed point, so those tests would be
# measuring rounding. Run the suite in float64. The library itself works in
# either, because every matmul feeding a Mahalanobis distance is pinned to
# Precision.HIGHEST, which is what stops an H200 from silently computing the
# quadratic form in TF32 (see the comment at the top of mkmeans_dpd/dpd.py).
jax.config.update("jax_enable_x64", True)


def elliptical(seed, n_per=200, k=3, p=2, spread=12.0):
    """K elliptical Gaussian clusters with different, correlated covariances."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(-spread, spread, size=(k, p))
    covs, parts, labels = [], [], []
    for j in range(k):
        a = rng.normal(size=(p, p)) * 0.6
        cov = a @ a.T + np.eye(p)
        covs.append(cov)
        parts.append(rng.multivariate_normal(means[j], cov, size=n_per))
        labels.append(np.full(n_per, j))
    return (
        np.concatenate(parts),
        np.concatenate(labels),
        means,
        np.stack(covs),
    )


@pytest.fixture
def small_cluster():
    """One well-behaved Gaussian sample, for single-cluster estimator tests."""
    rng = np.random.default_rng(7)
    cov = np.array([[2.0, 0.9], [0.9, 1.5]])
    return rng.multivariate_normal([1.0, -2.0], cov, size=400)
