"""Proposition 3.1 and Theorem 3.2, the paper's main algorithmic claim."""

import jax.numpy as jnp
import numpy as np
import pytest

from mkmeans_dpd import dc_mkmeans_dpd, mkmeans_dpd
from tests.conftest import elliptical

# Q is a sum of a few hundred float64 terms of comparable size that partly
# cancel, so "non-increasing" can only be checked to within the arithmetic's own
# noise. 1e-11 of the current value is about four orders above float64 epsilon
# and about eight below the decreases the algorithm actually makes.
_NOISE = 1e-11


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_dc_objective_is_non_increasing_at_every_half_step(seed, alpha):
    """Proposition 3.1, Equation (3.14).

    The trace holds Q after each half-step: the assignment of Step 4 and the
    update of Step 5, separately. Both are proved not to increase, part (i) and
    part (ii) of the proof, so checking only the per-iteration values would miss
    a violation in one half cancelling out in the other.
    """
    x, _, means, covs = elliptical(seed, n_per=150)
    res = dc_mkmeans_dpd(x, 3, alpha=alpha, centers=means, covariances=covs)
    q = np.asarray(res.objective)
    rises = np.diff(q) > _NOISE * np.abs(q[:-1])
    assert not rises.any(), (
        f"Q rose at half-steps {np.flatnonzero(rises).tolist()}: "
        f"{np.diff(q)[rises].tolist()}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_dc_partition_reaches_a_fixed_point_in_finitely_many_steps(seed):
    """Theorem 3.2: C^(t) = C^(T) for all t >= T, at some finite T.

    ``label_fixed_at`` is the first round after which the partition never moved
    again, so it being set at all is the finite-termination claim; we also check
    the history directly instead of trusting the bookkeeping.
    """
    x, _, means, covs = elliptical(seed, n_per=150)
    res = dc_mkmeans_dpd(x, 3, alpha=0.3, centers=means, covariances=covs, max_iter=200)

    assert res.label_fixed_at is not None, "partition still moving at max_iter"
    assert res.converged
    hist = np.asarray(res.label_history)
    t = res.label_fixed_at
    assert np.all(hist[t + 1 :] == hist[t + 1]), "partition moved after label_fixed_at"
    assert t < res.n_iter <= 200


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_dc_fixed_point_is_stable_under_both_steps(seed):
    """The "simultaneously stable under Step 4 and Step 5" part of Theorem 3.2.

    Restarting from the returned parameters must reproduce the same partition
    on the first assignment and leave the parameters alone.
    """
    x, _, means, covs = elliptical(seed, n_per=150)
    res = dc_mkmeans_dpd(x, 3, alpha=0.3, centers=means, covariances=covs)
    again = dc_mkmeans_dpd(
        x, 3, alpha=0.3, centers=res.centers, covariances=res.covariances, max_iter=3
    )
    assert bool(jnp.all(again.labels == res.labels))
    np.testing.assert_allclose(
        np.asarray(again.centers), np.asarray(res.centers), atol=1e-6
    )


def test_the_assignment_half_step_alone_never_increases_q():
    """Part (i) of the proof of Proposition 3.1, isolated.

    Entries 1, 3, 5, ... of the trace are the assignment half-steps; each is
    compared against the value immediately before it, with the parameters held
    fixed across the comparison.
    """
    x, _, means, covs = elliptical(2, n_per=150)
    q = np.asarray(
        dc_mkmeans_dpd(x, 3, alpha=0.4, centers=means, covariances=covs).objective
    )
    assign = q[1::2] - q[0:-1:2]
    update = q[2::2] - q[1:-1:2]
    assert np.all(assign <= _NOISE * np.abs(q[0:-1:2]))
    assert np.all(update <= _NOISE * np.abs(q[1:-1:2]))


def test_mkmeans_dpd_has_no_such_guarantee():
    """Section 3.3: Algorithm 2's assignment and update target different things.

    This is not a claim that Algorithm 2 always fails, only that it is not
    entitled to the theorem. It runs and it terminates; that is all we assert.
    We do check the two algorithms genuinely differ, so the DC variant is not
    silently the same code path.
    """
    x, _, means, covs = elliptical(4, n_per=150)
    a2 = mkmeans_dpd(x, 3, alpha=0.6, centers=means, covariances=covs)
    a3 = dc_mkmeans_dpd(x, 3, alpha=0.6, centers=means, covariances=covs)
    assert a2.n_iter >= 1 and a3.n_iter >= 1
    # Contaminate to pull the two assignment rules apart.
    rng = np.random.default_rng(0)
    dirty = np.concatenate([x, rng.uniform(-40, 40, size=(60, 2))])
    b2 = mkmeans_dpd(dirty, 3, alpha=0.6, centers=means, covariances=covs)
    b3 = dc_mkmeans_dpd(dirty, 3, alpha=0.6, centers=means, covariances=covs)
    assert not bool(jnp.all(b2.labels == b3.labels))


def test_the_eigenvalue_floor_is_actually_enforced():
    """The delta of Theorem 3.2, without which Q is unbounded below.

    Every covariance the fitter returns must have all eigenvalues at or above
    ``cov_floor``, on data built to tempt a cluster into collapsing: a tight
    knot of nearly collinear points next to two ordinary clusters.
    """
    rng = np.random.default_rng(1)
    knot = np.stack([np.linspace(0, 0.01, 40), np.linspace(0, 0.01, 40)], axis=1)
    knot = knot + rng.normal(scale=1e-7, size=knot.shape)
    x = np.concatenate(
        [
            knot,
            rng.normal(loc=[10, 0], size=(200, 2)),
            rng.normal(loc=[0, 10], size=(200, 2)),
        ]
    )
    floor = 1e-4
    res = dc_mkmeans_dpd(
        x,
        3,
        alpha=0.3,
        centers=np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        cov_floor=floor,
    )
    evals = np.linalg.eigvalsh(np.asarray(res.covariances))
    assert evals.min() >= floor * (1 - 1e-9)
    assert np.isfinite(np.asarray(res.objective)).all()
