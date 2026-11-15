import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coop_agents.consensus import (
    majority,
    majority_loo,
    simplicial,
    voter,
    weighted_average,
)
from coop_agents.losses import confidence


def test_majority_of_three_is_class_zero():
    # Three agents predicting classes [0, 0, 1].
    probs = jnp.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    cons = majority(probs)
    assert cons.shape == (2,)
    np.testing.assert_allclose(cons, jnp.array([1.0, 0.0]))
    assert int(jnp.argmax(cons)) == 0


def test_majority_loo_tie_breaks_to_smallest_index():
    """Paper: ties take the smallest class index.

    Agents predict [0, 0, 1]. The class-1 voter sees majority 0. Each class-0
    voter sees a 0-vs-1 tie, which breaks to class 0.
    """
    probs = jnp.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    cons = majority_loo(probs)
    assert cons.shape == (3, 2)
    # Agent 2 (class-1 vote) sees two class-0 votes.
    np.testing.assert_allclose(cons[2], jnp.array([1.0, 0.0]))
    # Class-0 agents see a tie; smallest-index tie-break -> class 0.
    np.testing.assert_allclose(cons[0], jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(cons[1], jnp.array([1.0, 0.0]))


def test_voter_fixed_key_is_deterministic_keys_can_differ():
    probs = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    k0 = jax.random.PRNGKey(0)
    a = voter(probs, k0)
    b = voter(probs, k0)
    np.testing.assert_allclose(a, b)

    seen = []
    for seed in range(32):
        seen.append(tuple(np.asarray(voter(probs, jax.random.PRNGKey(seed))).tolist()))
    assert len(set(seen)) >= 2


def test_weighted_average_of_identical_softmaxes():
    p = jnp.array([0.2, 0.5, 0.3])
    probs = jnp.stack([p, p, p], axis=0)
    w = jnp.array([0.1, 0.2, 0.7])
    got = weighted_average(probs, w)
    np.testing.assert_allclose(got, p, rtol=1e-6, atol=1e-6)
    w_eq = jnp.ones((3,))
    np.testing.assert_allclose(weighted_average(probs, w_eq), p)


def test_simplicial_matches_weighted_average_and_loo_drops_self():
    probs = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    w = confidence(probs)
    np.testing.assert_allclose(
        simplicial(probs, w, loo=False),
        weighted_average(probs, w),
    )
    loo = simplicial(probs, w, loo=True)
    assert loo.shape == probs.shape
    # Agent 0 should not equal the full consensus when others disagree.
    assert not jnp.allclose(loo[0], simplicial(probs, w, loo=False))
