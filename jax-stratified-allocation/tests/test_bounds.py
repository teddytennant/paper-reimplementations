"""Proposition 11: the Serfling bound over a finite candidate set."""

import jax
import jax.numpy as jnp
import pytest

from stratified_allocation import class_counts, serfling_uniform_bound
from stratified_allocation.designs import class_index_sets, stratified_sample


def _candidates(key, n, n_candidates, bound_lambda):
    """M candidate loss vectors, each row bounded by Lambda in absolute value."""
    u = jax.random.uniform(key, (n_candidates, n), dtype=jnp.float64, minval=-1.0, maxval=1.0)
    return bound_lambda * u


@pytest.mark.parametrize("k0, k1", [(20, 20), (50, 10), (10, 40)])
@pytest.mark.parametrize("epsilon", [0.5, 0.1, 0.01])
def test_bound_holds_in_simulation(k0, k1, epsilon):
    """Monte Carlo tail probability of the supremum, against the stated 1 - epsilon."""
    n, n1, n_candidates, lam = 400, 60, 8, 1.0
    labels = jnp.concatenate([jnp.zeros(n - n1, jnp.int32), jnp.ones(n1, jnp.int32)])
    counts = class_counts(labels)
    idx0, idx1 = class_index_sets(labels)
    cand = _candidates(jax.random.key(41), n, n_candidates, lam)
    totals = jnp.sum(cand, axis=1)

    def sup_dev(key):
        s0, s1 = stratified_sample(key, idx0, idx1, k0, k1)
        est = (n - n1) / k0 * jnp.sum(cand[:, s0], axis=1) + n1 / k1 * jnp.sum(
            cand[:, s1], axis=1
        )
        return jnp.max(jnp.abs(est - totals))

    reps = 20_000
    keys = jax.random.split(jax.random.key(42), reps)
    devs = jax.jit(jax.vmap(sup_dev))(keys)

    bound = float(
        serfling_uniform_bound(counts, jnp.array([k0, k1], jnp.float64), n_candidates, lam, epsilon)
    )
    assert float(jnp.mean(devs > bound)) <= epsilon
    # And the bound is not vacuously large: over 20k draws the worst supremum gets to
    # more than a third of it, so the test would notice a bound inflated by 3x.
    assert float(jnp.max(devs)) > 0.3 * bound


def test_bound_matches_the_written_formula():
    counts = jnp.array([900.0, 100.0])
    alloc = jnp.array([60.0, 40.0])
    lam, m, eps = 2.5, 16, 0.05
    expected = lam * sum(
        float(counts[c])
        * float(jnp.sqrt(2.0 * (1.0 - (alloc[c] - 1.0) / counts[c]) / alloc[c] * jnp.log(2 * m / eps)))
        for c in (0, 1)
    )
    got = float(serfling_uniform_bound(counts, alloc, m, lam, eps))
    assert abs(got - expected) < 1e-9 * expected


def test_bound_shrinks_like_one_over_sqrt_K():
    """The O(K_c^{-1/2}) rate of Equation (16), with the N_c weighting of Equation (11)."""
    # Big strata, so the (1 - (K_c - 1)/N_c) factor stays near 1 and only the K_c^{-1/2}
    # is visible. At small N_c that factor dominates and the rate looks faster than
    # K_c^{-1/2}, which is the sampling-without-replacement gain, not a bug.
    counts = jnp.array([90_000.0, 10_000.0])
    scaled = [
        float(serfling_uniform_bound(counts, jnp.array([4.0 * k, k]), 10, 1.0, 0.1))
        for k in (4.0, 16.0, 64.0)
    ]
    assert scaled[0] > scaled[1] > scaled[2]
    # Quadrupling K_c halves the bound.
    assert 1.99 < scaled[0] / scaled[1] < 2.01
    assert 1.99 < scaled[1] / scaled[2] < 2.01

    small = [
        float(serfling_uniform_bound(jnp.array([900.0, 100.0]), jnp.array([4.0 * k, k]), 10, 1.0, 0.1))
        for k in (16.0, 64.0)
    ]
    assert small[0] / small[1] > 2.01


def test_bound_grows_with_the_candidate_count_and_the_confidence():
    base = serfling_uniform_bound(jnp.array([900.0, 100.0]), jnp.array([50.0, 50.0]), 10, 1.0, 0.1)
    more = serfling_uniform_bound(jnp.array([900.0, 100.0]), jnp.array([50.0, 50.0]), 1000, 1.0, 0.1)
    tighter = serfling_uniform_bound(
        jnp.array([900.0, 100.0]), jnp.array([50.0, 50.0]), 10, 1.0, 0.001
    )
    assert float(more) > float(base)
    assert float(tighter) > float(base)
