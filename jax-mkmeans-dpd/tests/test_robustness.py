"""Robustness stated as a measurement: how far do the centers move under
contamination, for the DPD methods against the two classical baselines.

This is the claim of Section 4.2, that the DPD estimation of (mu_k, Sigma_k)
resists outliers where the maximum likelihood updates of Algorithms 1 and
classical K-means do not.
"""

from itertools import permutations

import jax.numpy as jnp
import numpy as np
import pytest

from mkmeans_dpd import dc_mkmeans_dpd, kmeans, mkmeans, mkmeans_dpd
from tests.conftest import elliptical

SEEDS = [0, 1, 2, 3, 4, 5, 6]
CONTAMINATION = 0.08


def _displacement(clean, dirty):
    """Largest center movement, minimized over the K! ways to match them up."""
    d = np.linalg.norm(np.asarray(clean)[:, None, :] - np.asarray(dirty)[None], axis=-1)
    return min(
        max(d[i, perm[i]] for i in range(len(d)))
        for perm in permutations(range(len(d)))
    )


def _contaminate(x, rng, frac=CONTAMINATION):
    """Append far outliers: a fixed number of points well outside the data."""
    n_out = round(frac * len(x))
    span = x.max(0) - x.min(0)
    out = x.mean(0) + rng.uniform(6.0, 9.0, size=(n_out, x.shape[1])) * span
    return np.concatenate([x, out])


def _fits(seed, alpha):
    """Fit every method twice, clean and contaminated, from the same seed points.

    Both fits of a given method start from the true means and covariances, so
    the only difference between them is the contamination. Randomized seeding
    would mix initialization variance into the measurement.
    """
    x, _, means, covs = elliptical(seed, n_per=200)
    dirty = _contaminate(x, np.random.default_rng(1000 + seed))
    out = {}
    for name, fit in (
        ("kmeans", lambda d: kmeans(d, 3, centers=means).centers),
        ("mkmeans", lambda d: mkmeans(d, 3, centers=means).centers),
        (
            "mkmeans_dpd",
            lambda d: (
                mkmeans_dpd(d, 3, alpha=alpha, centers=means, covariances=covs).centers
            ),
        ),
        (
            "dc_mkmeans_dpd",
            lambda d: (
                dc_mkmeans_dpd(
                    d, 3, alpha=alpha, centers=means, covariances=covs
                ).centers
            ),
        ),
    ):
        out[name] = _displacement(fit(x), fit(dirty))
    return out


@pytest.fixture(scope="module")
def displacements():
    rows = [_fits(s, alpha=0.5) for s in SEEDS]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def test_dpd_centers_barely_move_under_contamination(displacements):
    """8% of far outliers must not move a DPD center by more than a fraction of
    a within-cluster standard deviation. The clusters here have unit-ish scale,
    so 0.3 is a real bound and not a formality."""
    for name in ("mkmeans_dpd", "dc_mkmeans_dpd"):
        med = float(np.median(displacements[name]))
        assert med < 0.3, f"{name} median displacement {med:.3f}"


def test_classical_kmeans_centers_move_far_more(displacements):
    """The comparison the paper makes. Asserted on the median over 7 seeds, not
    on any single run, since one draw of outliers can happen to land harmlessly."""
    dpd = float(np.median(displacements["dc_mkmeans_dpd"]))
    for name in ("kmeans", "mkmeans"):
        med = float(np.median(displacements[name]))
        assert med > 10 * dpd, f"{name} median displacement {med:.3f} vs DPD {dpd:.3f}"


def test_estimator_bias_falls_as_alpha_rises():
    """Section 3.1: "as alpha increases, the robustness measure enhances".

    Measured on the estimator itself rather than through a clustering, so the
    result is not confounded by which points ended up in which cluster. One
    Gaussian sample at the origin plus 5% of points sitting at (5, 5): the MDPDE
    of the mean and of the covariance must both walk back toward the truth as
    alpha grows. At alpha = 1e-6 this is the MLE and carries the full bias.

    The contamination is deliberately moderate. Put the outliers far enough out
    and every alpha above zero discards them completely, and the ordering says
    nothing.
    """
    from mkmeans_dpd import mdpde, spec_from_cov

    rng = np.random.default_rng(0)
    x = np.concatenate(
        [rng.normal(size=(300, 2)), np.full((15, 2), 5.0) + rng.normal(0, 0.2, (15, 2))]
    )
    x = jnp.asarray(x)
    resp = jnp.ones((x.shape[0], 1))

    bias, det = [], []
    for alpha in (1e-6, 0.05, 0.1, 0.2):
        mu, spec = mdpde(
            x,
            jnp.mean(x, axis=0)[None],
            spec_from_cov(jnp.cov(x.T, bias=True)[None], 1e-12),
            resp,
            alpha,
            1e-12,
            n_inner=300,
        )
        bias.append(float(jnp.linalg.norm(mu[0])))
        det.append(float(jnp.linalg.det(spec.sigma[0])))

    assert bias == sorted(bias, reverse=True), bias
    assert det == sorted(det, reverse=True), det
    # The MLE end carries the full contamination; the robust end does not.
    assert bias[0] > 0.3 and bias[-1] < 0.1
    assert det[0] > 2.5 and abs(det[-1] - 1.0) < 0.15  # true covariance is I


def test_dpd_recovers_the_true_centers_under_contamination():
    """Displacement is only half the claim; the centers also have to be right.

    Fit on contaminated data and compare against the generating means.
    """
    errs = []
    for seed in SEEDS:
        x, _, means, covs = elliptical(seed, n_per=200)
        dirty = _contaminate(x, np.random.default_rng(1000 + seed))
        res = dc_mkmeans_dpd(dirty, 3, alpha=0.5, centers=means, covariances=covs)
        errs.append(_displacement(means, res.centers))
    assert float(np.median(errs)) < 0.3
