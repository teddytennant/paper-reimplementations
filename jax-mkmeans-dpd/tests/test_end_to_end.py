"""Data loading, initialization and whole-pipeline behavior."""

import numpy as np
import pytest

from mkmeans_dpd import (
    cluster_accuracy,
    core_init,
    dc_mkmeans_dpd,
    kmeans,
    load_iris,
    make_simulated,
    mkmeans,
    mkmeans_dpd,
)
from mkmeans_dpd.init import chi2_quantile_99
from tests.conftest import elliptical


def test_iris_is_the_dataset_of_section_5_1():
    x, y = load_iris()
    assert x.shape == (150, 4)
    assert sorted(np.bincount(y).tolist()) == [50, 50, 50]
    # Fisher's first row, as a check that the vendored copy is the real one.
    np.testing.assert_allclose(x[0], [5.1, 3.5, 1.4, 0.2])
    assert x.min() > 0


def test_simulated_data_matches_section_4_2():
    """750 points in 3 groups of 150 / 300 / 300, plus 30 appended outliers."""
    x, y = make_simulated(seed=0)
    assert x.shape == (780, 2)
    counts = [int((y == j).sum()) for j in range(3)]
    assert counts == [150, 300, 300]
    assert int((y == -1).sum()) == 30
    # Group 1's covariance has off-diagonal 1.4, group 2's has -1.4.
    assert np.cov(x[y == 0].T)[0, 1] > 0.8
    assert np.cov(x[y == 1].T)[0, 1] < -0.8


def test_chi_square_quantile_approximation():
    """Wilson-Hilferty against the exact 99% points of chi-square, p = 1 to 6."""
    exact = {1: 6.6349, 2: 9.2103, 3: 11.3449, 4: 13.2767, 5: 15.0863, 6: 16.8119}
    for p, want in exact.items():
        # p = 1 is the worst case at 0.74%; every other p is inside 0.25%.
        assert chi2_quantile_99(p) == pytest.approx(want, rel=1e-2)


def test_core_init_finds_the_true_centers():
    """Section 3.2.1 should land near each generating mean, not on a stray point."""
    import jax

    x, _, means, _ = elliptical(3, n_per=250)
    centers, covs = core_init(x, 3, alpha=0.2, key=jax.random.PRNGKey(1))
    assert centers.shape == (3, 2) and covs.shape == (3, 2, 2)
    d = np.linalg.norm(np.asarray(centers)[:, None] - means[None], axis=-1)
    assert np.all(d.min(axis=1) < 2.0)
    assert len(set(np.argmin(d, axis=1).tolist())) == 3, "two seeds hit one cluster"


def test_kmeans_recovers_separated_spherical_clusters():
    rng = np.random.default_rng(0)
    means = np.array([[0.0, 0.0], [15.0, 0.0], [0.0, 15.0]])
    x = np.concatenate([m + rng.normal(size=(120, 2)) for m in means])
    y = np.repeat(np.arange(3), 120)
    res = kmeans(x, 3, centers=means)
    assert cluster_accuracy(y, res.labels, 3) == 1.0


def test_mahalanobis_beats_euclidean_on_elliptical_clusters():
    """The motivation for Algorithm 1 in Section 2: Euclidean K-means cuts
    stretched, correlated clusters along the wrong axis."""
    rng = np.random.default_rng(0)
    cov = np.array([[9.0, 8.4], [8.4, 9.0]])
    means = np.array([[0.0, 0.0], [3.0, -3.0]])
    x = np.concatenate([rng.multivariate_normal(m, cov, 400) for m in means])
    y = np.repeat(np.arange(2), 400)
    km = cluster_accuracy(y, kmeans(x, 2, centers=means).labels, 2)
    mk = cluster_accuracy(y, mkmeans(x, 2, centers=means).labels, 2)
    assert mk > km + 0.05


@pytest.mark.parametrize("fitter", [mkmeans_dpd, dc_mkmeans_dpd])
def test_iris_accuracy_matches_the_paper(fitter):
    """Table 3 reports 0.967 for both MK-means DPD and DC-MK-means DPD at
    alpha = 0.2. Started from the Section 3.2.1 initialization."""
    import jax

    x, y = load_iris()
    res = fitter(x, 3, alpha=0.2, key=jax.random.PRNGKey(0), n_init=5)
    assert cluster_accuracy(y, res.labels, 3) >= 0.92


def test_simulated_accuracy_beats_classical_kmeans():
    """Table 2: K-means 0.788, DC-MK-means DPD 0.955 on the Section 4.2 data.
    Outliers are excluded from the accuracy, as they have no true group."""
    import jax

    x, y = make_simulated(seed=0)
    dc = dc_mkmeans_dpd(x, 3, alpha=0.2, key=jax.random.PRNGKey(1), n_init=3)
    km = kmeans(x, 3, key=jax.random.PRNGKey(0))
    assert cluster_accuracy(y, dc.labels, 3) > 0.9
    assert cluster_accuracy(y, dc.labels, 3) > cluster_accuracy(y, km.labels, 3) + 0.1


def test_empty_cluster_does_not_break_the_fit():
    """Ask for more clusters than there are groups; the extra one empties out
    and must leave the objective finite rather than dividing by n_k = 0."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(100, 2)), rng.normal(loc=20, size=(100, 2))])
    centers = np.array([[0.0, 0.0], [20.0, 20.0], [200.0, 200.0]])
    res = dc_mkmeans_dpd(x, 3, alpha=0.3, centers=centers)
    assert np.isfinite(np.asarray(res.objective)).all()
    assert np.isfinite(np.asarray(res.covariances)).all()


def test_demo_runs(capsys):
    from mkmeans_dpd.demo import run

    x, y = load_iris()
    run(x, y, 3, 0.2, 0, 0.1, "Iris, Section 5.1", n_restarts=1)
    out = capsys.readouterr().out
    assert "DC-MK-means DPD" in out and "CHI_trim" in out
    assert "Theorem 3.2" in out
