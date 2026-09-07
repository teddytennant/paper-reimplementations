"""Section 6.5: the empirical-distribution rank map and its conformal correction."""

import jax
import jax.numpy as jnp

from stratified_allocation import (
    average_precision,
    conformal_rank_map,
    rank_map,
    roc_auc,
)


def _fold(key, n=400, prevalence=0.2):
    k1, k2 = jax.random.split(key)
    labels = (jax.random.uniform(k1, (n,), dtype=jnp.float64) < prevalence).astype(jnp.int32)
    scores = jax.random.normal(k2, (n,), dtype=jnp.float64) + 1.2 * labels
    return labels, scores


def test_rank_map_is_nondecreasing():
    val = jax.random.normal(jax.random.key(50), (5000,), dtype=jnp.float64)
    q = jnp.sort(jax.random.normal(jax.random.key(51), (300,), dtype=jnp.float64))
    u = rank_map(val, q)
    assert bool(jnp.all(jnp.diff(u) >= 0.0))


def test_proposition_13_rank_invariance():
    """With u injective on the fold's test scores, average precision and ROC-AUC are
    unchanged by the map, exactly.

    Injectivity is checked rather than assumed: it is the hypothesis of Proposition 13,
    and without it the map creates ties and the two quantities differ.
    """
    labels, scores = _fold(jax.random.key(52))
    # One validation score in each gap between consecutive test scores, so u is
    # injective on the fold by construction rather than by luck. Drawing a large
    # validation sample instead leaves a real chance of a collision: with 400 test
    # scores and 200k validation scores it is about one in three.
    order = jnp.sort(scores)
    val = (order[:-1] + order[1:]) / 2.0
    u = rank_map(val, scores)
    assert int(jnp.unique(u).shape[0]) == int(jnp.unique(scores).shape[0])

    assert jnp.allclose(average_precision(labels, u), average_precision(labels, scores), rtol=0, atol=1e-12)
    assert jnp.allclose(roc_auc(labels, u), roc_auc(labels, scores), rtol=0, atol=1e-12)


def test_without_injectivity_the_two_differ():
    """The escape clause of Proposition 13: the difference is only through the ties u
    creates."""
    labels, scores = _fold(jax.random.key(54))
    val = jnp.array([-0.5, 0.0, 0.5], dtype=jnp.float64)
    u = rank_map(val, scores)
    assert int(jnp.unique(u).shape[0]) < int(jnp.unique(scores).shape[0])
    assert abs(float(roc_auc(labels, u)) - float(roc_auc(labels, scores))) > 1e-3


def test_corrected_map_is_a_valid_p_value_under_exchangeability():
    """Proposition 15: Pr{utilde(Q) <= alpha} <= alpha when the scores are exchangeable.

    Draw n_V + 1 exchangeable scores, hold one out as the test score, and check the
    coverage over many draws. The bound is on the exact discrete distribution, so this
    only has to hold up to Monte Carlo error, checked at three levels.
    """
    n_v, reps = 19, 200_000
    keys = jax.random.split(jax.random.key(55), reps)

    def one(key):
        z = jax.random.normal(key, (n_v + 1,), dtype=jnp.float64)
        return conformal_rank_map(z[:n_v], z[n_v:])[0]

    p = jax.jit(jax.vmap(one))(keys)
    for alpha in (0.05, 0.1, 0.25, 0.5):
        rate = float(jnp.mean(p <= alpha))
        se = (alpha * (1 - alpha) / reps) ** 0.5
        assert rate <= alpha + 5.0 * se


def test_uncorrected_map_can_exceed_alpha():
    """Why Equation (19) has the correction at all: u itself is not a p-value."""
    n_v, reps = 19, 200_000
    keys = jax.random.split(jax.random.key(56), reps)

    def one(key):
        z = jax.random.normal(key, (n_v + 1,), dtype=jnp.float64)
        return rank_map(z[:n_v], z[n_v:])[0]

    p = jax.jit(jax.vmap(one))(keys)
    # u takes values k/19, so the level has to sit just above one of them for the
    # overshoot to show; at alpha = 0.06 the event is count in {0, 1}, probability 0.1.
    for alpha in (0.06, 0.16):
        rate = float(jnp.mean(p <= alpha))
        se = (alpha * (1 - alpha) / reps) ** 0.5
        assert rate > alpha + 5.0 * se
