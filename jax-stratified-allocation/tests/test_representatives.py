"""Proposition 7: deterministic cluster representatives admit no unbiasedness statement."""

import itertools

import jax
import jax.numpy as jnp

from stratified_allocation import (
    cluster_representatives,
    kmeans_labels,
    representative_bias,
    representative_estimate,
    weighted_losses,
)


def _small_clustered_population(offset=0.0):
    """Twelve rows in three clusters of four, with losses that vary inside a cluster.

    The losses are written down rather than drawn, so the enumeration below is the same
    on every backend.
    """
    assignment = jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    labels = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], jnp.int32)
    raw = jnp.array(
        [0.4, -1.2, 0.9, 2.1, -0.3, 0.7, 1.6, -2.0, 0.2, 1.1, -0.8, 0.5], jnp.float64
    ) + offset
    return assignment, labels, weighted_losses(raw, labels)


def test_bias_identity_holds_for_every_choice_of_representative():
    """Equation (10) by exact enumeration.

    Three clusters of four rows is 4^3 = 64 possible representative sets. For each one,
    the identity is checked against the estimate minus the estimand computed directly.
    """
    assignment, _, losses = _small_clustered_population()
    total = jnp.sum(losses)
    members = [jnp.where(assignment == c)[0] for c in range(3)]

    for choice in itertools.product(*[range(len(m)) for m in members]):
        reps = jnp.array([int(members[c][choice[c]]) for c in range(3)])
        est = representative_estimate(losses, assignment, reps, 3)
        identity = representative_bias(losses, assignment, reps, 3)
        assert jnp.allclose(est - total, identity, rtol=1e-12, atol=1e-12)


def test_identity_vanishes_exactly_when_losses_are_constant_within_clusters():
    """The "only if" half of Proposition 7."""
    assignment = jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    constant = jnp.array([2.0, 2.0, 2.0, -1.0, -1.0, -1.0, 5.0, 5.0, 5.0])
    for reps in ([0, 3, 6], [1, 4, 7], [2, 5, 8]):
        b = representative_bias(constant, assignment, jnp.array(reps), 3)
        assert abs(float(b)) < 1e-12

    varying = constant.at[0].set(2.5)
    b = representative_bias(varying, assignment, jnp.array([0, 3, 6]), 3)
    assert abs(float(b)) > 0.1


def test_kmeans_representative_design_is_biased_here():
    """A concrete counterexample, because no general unbiasedness claim is available.

    Features are three tight blobs, so the clustering is the obvious one, but the loss
    inside each blob is a linear function of a coordinate the clustering has collapsed.
    The retained medoid sits at the blob centre and reports the cluster mean of that
    coordinate, while the loss is convex in it, so the estimate is short by a fixed
    amount no matter which run of the clustering produced it.
    """
    centres = jnp.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    offsets = jnp.linspace(-1.0, 1.0, 7)
    features = jnp.concatenate([c + jnp.stack([offsets, jnp.zeros(7)], 1) for c in centres])
    losses = features[:, 0] ** 2 + features[:, 1] ** 2

    # One starting centre per blob, so the partition is the intended one and does
    # not depend on how the random initialisation happened to fall.
    assignment = kmeans_labels(jax.random.key(2), features, 3, init=jnp.array([0, 7, 14]))
    assert sorted(int(jnp.sum(assignment == c)) for c in range(3)) == [7, 7, 7]
    reps = cluster_representatives(features, assignment, 3)
    est = representative_estimate(losses, assignment, reps, 3)
    bias = representative_bias(losses, assignment, reps, 3)

    assert jnp.allclose(est - jnp.sum(losses), bias, rtol=1e-12, atol=1e-10)
    # Each blob loses 7 * var(offsets) = 7 * (4/3) / ... , computed here rather than
    # asserted from a formula: what matters is that it is nowhere near zero.
    assert float(bias) < -5.0


def test_a_random_representative_would_be_unbiased_and_the_deterministic_one_is_not():
    """Where the bias comes from is the determinism, not the weighting.

    Averaging the identity over a uniformly drawn representative per cluster gives zero
    exactly, by enumeration. The medoid rule picks one specific term of that average and
    there is no reason for it to be the zero one.
    """
    assignment, _, losses = _small_clustered_population(offset=0.3)
    members = [jnp.where(assignment == c)[0] for c in range(3)]
    biases = []
    for choice in itertools.product(*[range(len(m)) for m in members]):
        reps = jnp.array([int(members[c][choice[c]]) for c in range(3)])
        biases.append(float(representative_bias(losses, assignment, reps, 3)))
    assert abs(sum(biases) / len(biases)) < 1e-11
    assert max(abs(b) for b in biases) > 0.5
