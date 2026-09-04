"""The internal indices of Section 4.1, classical and robust.

Each of the four is first checked against a configuration whose value can be
worked out by hand, then against the same data with one gross outlier added.
The point of Equations (4.7) and (4.8) is that the second check barely moves
them while DBI (4.3) and CHI (4.4) swing hard.
"""

import numpy as np
import pytest

from mkmeans_dpd import (
    calinski_harabasz,
    davies_bouldin,
    median_davies_bouldin,
    trimmed_calinski_harabasz,
)

# Two clusters of two points each: cluster 0 at 0 +- s along x, cluster 1 at
# d +- s. Centroids 0 and d, mean and median distance to centroid both s.
S, D = 0.5, 6.0
HAND = np.array([[-S, 0.0], [S, 0.0], [D - S, 0.0], [D + S, 0.0]])
HAND_LABELS = np.array([0, 0, 1, 1])


def test_dbi_matches_the_hand_calculation():
    """DB = (1/K) sum_i max_j (sigma_i + sigma_j)/d(c_i, c_j) = 2s/d."""
    assert davies_bouldin(HAND, HAND_LABELS, 2) == pytest.approx(2 * S / D)


def test_median_dbi_matches_the_hand_calculation():
    """With two symmetric points per cluster the median center is the mean and
    the median distance is the mean distance, so DBI_M must equal DBI here.
    That pins the definition of Equation (4.7) rather than only its behavior."""
    assert median_davies_bouldin(HAND, HAND_LABELS, 2) == pytest.approx(2 * S / D)


def test_chi_matches_the_hand_calculation():
    """tr(B) = 2 (d/2)^2 * 2 = d^2, tr(W) = 4 s^2, CH = d^2/(4s^2) * (n-K)/(K-1)."""
    expect = D**2 / (4 * S**2) * (4 - 2) / (2 - 1)
    assert calinski_harabasz(HAND, HAND_LABELS, 2) == pytest.approx(expect)


def test_trimmed_chi_equals_chi_when_nothing_is_trimmed():
    """Every within-cluster distance here is s, so the (1 - trim) quantile is s
    and no point is dropped. Equation (4.8) then reduces to Equation (4.4)."""
    expect = D**2 / (4 * S**2) * (4 - 2) / (2 - 1)
    assert trimmed_calinski_harabasz(HAND, HAND_LABELS, 2, trim=0.1) == pytest.approx(
        expect
    )


def test_trimming_preserves_the_variance_decomposition():
    """The stated reason for trimming CHI rather than taking medians:
    tr(B_trim) + tr(W_trim) = tr(T_trim) has to hold exactly on the retained
    sample, which it does only because the centers stay ordinary sample means.
    """
    x, labels = _separated()
    trim = 0.2
    keep_x, keep_lab = [], []
    for j in range(3):
        pts = x[labels == j]
        d = np.linalg.norm(pts - pts.mean(0), axis=1)
        sel = d <= np.quantile(d, 1 - trim)
        keep_x.append(pts[sel])
        keep_lab.append(np.full(sel.sum(), j))
    xt, lt = np.concatenate(keep_x), np.concatenate(keep_lab)

    grand = xt.mean(0)
    centers = np.stack([xt[lt == j].mean(0) for j in range(3)])
    nk = np.array([(lt == j).sum() for j in range(3)])
    tr_b = float((nk * ((centers - grand) ** 2).sum(1)).sum())
    tr_w = float(((xt - centers[lt]) ** 2).sum())
    tr_t = float(((xt - grand) ** 2).sum())
    assert tr_b + tr_w == pytest.approx(tr_t, rel=1e-10)


def _separated(seed=0, n=100):
    """Three tight, cleanly separated spherical clusters."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    x = np.concatenate([m + rng.normal(scale=0.5, size=(n, 2)) for m in means])
    labels = np.repeat(np.arange(3), n)
    return x, labels


def _with_outlier(x, labels, at=(600.0, 600.0)):
    """The same data plus a single gross outlier, put in cluster 0."""
    return np.vstack([x, np.array(at)]), np.append(labels, 0)


def test_indices_agree_that_separated_clusters_are_good():
    x, labels = _separated()
    assert davies_bouldin(x, labels, 3) < 0.1
    assert median_davies_bouldin(x, labels, 3) < 0.1
    assert calinski_harabasz(x, labels, 3) > 5000
    assert trimmed_calinski_harabasz(x, labels, 3) > 5000


def test_one_gross_outlier_wrecks_the_classical_indices():
    """One point in 301 moves DBI and CHI by more than an order of magnitude."""
    x, labels = _separated()
    xo, lo = _with_outlier(x, labels)
    assert davies_bouldin(xo, lo, 3) > 10 * davies_bouldin(x, labels, 3)
    assert calinski_harabasz(xo, lo, 3) < 0.1 * calinski_harabasz(x, labels, 3)


def test_the_robust_indices_barely_notice_it():
    """Equations (4.7) and (4.8) under the same single outlier. The median
    center and median dispersion of a 101-point cluster shift by one rank, and
    a 10% trim removes the point outright."""
    x, labels = _separated()
    xo, lo = _with_outlier(x, labels)

    base_db = median_davies_bouldin(x, labels, 3)
    hit_db = median_davies_bouldin(xo, lo, 3)
    assert abs(hit_db - base_db) < 0.05 * base_db

    base_ch = trimmed_calinski_harabasz(x, labels, 3)
    hit_ch = trimmed_calinski_harabasz(xo, lo, 3)
    # The remaining few percent is the trim boundary moving, not the outlier:
    # cluster 0 grows from 100 points to 101, so a 10% trim keeps a different
    # number of them. CHI itself loses 99.98% of its value on the same data.
    assert abs(hit_ch - base_ch) < 0.07 * base_ch


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_robust_indices_move_less_than_the_classical_ones(seed):
    """The comparison, over several draws rather than the one above."""
    x, labels = _separated(seed)
    xo, lo = _with_outlier(x, labels)

    def rel(f, *a, **kw):
        b = f(x, labels, 3, *a, **kw)
        return abs(f(xo, lo, 3, *a, **kw) - b) / abs(b)

    assert rel(median_davies_bouldin) < 0.01 * rel(davies_bouldin)
    assert rel(trimmed_calinski_harabasz) < 0.1 * rel(calinski_harabasz)


def test_median_dbi_uses_the_median_center_not_the_mean():
    """A skewed cluster separates the two definitions: the coordinate-wise
    median of Equation (4.7) sits with the bulk of the points, the centroid is
    dragged toward the tail, so the two indices disagree."""
    rng = np.random.default_rng(2)
    bulk = rng.normal(scale=0.3, size=(150, 2))
    tail = np.column_stack([rng.uniform(3, 9, 30), rng.normal(0, 0.3, 30)])
    x = np.concatenate([bulk, tail, rng.normal([25, 0], 0.3, (150, 2))])
    labels = np.concatenate([np.zeros(180, int), np.ones(150, int)])
    assert median_davies_bouldin(x, labels, 2) < 0.5 * davies_bouldin(x, labels, 2)
