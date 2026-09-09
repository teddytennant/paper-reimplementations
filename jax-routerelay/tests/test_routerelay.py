"""Lemma 1, eq 8 reuse test, near-miss construction, and eq 10."""

import math

import jax
import jax.numpy as jnp
import pytest

from routerelay.relay import (
    anchor_margin,
    build_ledger,
    miss_probability,
    miss_probability_bound,
    near_miss_sentinels,
    relay_layer,
    rescore_candidates,
    topk_routes,
)


jax.config.update("jax_enable_x64", True)


def _ranked_scores(n_q, n_k, dtype=jnp.float64):
    cols = jnp.arange(n_k, dtype=dtype)
    return jnp.broadcast_to((n_k - cols)[None, :], (n_q, n_k))


def test_lemma1_routes_identical_when_margin_beats_twice_eps():
    n_q, n_k, k = 6, 20, 5
    eps = jnp.float64(0.3)
    delta = jnp.float64(2.0) * eps + jnp.float64(0.4)
    scores_a = jnp.zeros((n_q, n_k), dtype=jnp.float64)
    scores_a = scores_a.at[:, :k].set(delta)
    routes_a = topk_routes(scores_a, k)
    assert jnp.all(jnp.sort(routes_a, axis=1) == jnp.arange(k)[None, :])
    margins = anchor_margin(scores_a, routes_a)
    assert jnp.all(margins > 2.0 * eps)

    scores_ell = scores_a.at[:, :k].add(-eps).at[:, k:].add(eps)
    assert float(jnp.max(jnp.abs(scores_ell - scores_a))) <= float(eps) + 1e-15
    routes_ell = topk_routes(scores_ell, k)
    assert jnp.all(jnp.sort(routes_ell, axis=1) == jnp.sort(routes_a, axis=1))


def test_eq8_reuse_when_alpha_beats_beta_else_refresh_matches_topk():
    n_q, n_k, k, m, r = 2, 16, 3, 2, 0
    scores_a = jnp.array(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * (n_k - 5),
            [5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * (n_k - 5),
        ],
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(0)
    ledger = build_ledger(scores_a, k, m, r, key)
    assert jnp.all(jnp.sort(ledger.routes, axis=1) == jnp.array([[0, 1, 2], [0, 1, 2]]))

    k_c = jnp.eye(n_k, dtype=jnp.float64)
    _, _, alpha_a, beta_a = rescore_candidates(
        scores_a, k_c, ledger.routes, ledger.sentinels
    )
    assert jnp.all(alpha_a > beta_a)

    scores_ell = scores_a.at[0, 3].set(3.5)
    ledger = ledger._replace(ages=jnp.ones((n_q,), dtype=jnp.int32))
    new, info = relay_layer(
        scores_ell,
        k_c,
        ledger,
        k=k,
        m=m,
        r=r,
        gamma=0.0,
        rng_key=key,
        hysteresis_horizon=1,
    )
    assert bool(info.reuse[1])
    assert not bool(info.reuse[0])
    assert jnp.all(new.routes[1] == ledger.routes[1])
    exact0 = jnp.sort(topk_routes(scores_ell[0:1], k), axis=1)[0]
    assert jnp.all(jnp.sort(new.routes[0]) == exact0)
    assert jnp.all(jnp.sort(new.routes[0]) == jnp.array([0, 1, 3]))


def test_near_miss_are_the_next_m_after_topk():
    n_q, n_k, k, m = 5, 24, 6, 4
    scores = _ranked_scores(n_q, n_k)
    routes = topk_routes(scores, k)
    near = near_miss_sentinels(scores, routes, m)
    expect_routes = jnp.broadcast_to(jnp.arange(k)[None, :], (n_q, k))
    expect_near = jnp.broadcast_to(jnp.arange(k, k + m)[None, :], (n_q, m))
    assert jnp.all(jnp.sort(routes, axis=1) == expect_routes)
    assert jnp.all(jnp.sort(near, axis=1) == expect_near)


@pytest.mark.parametrize("n_pool,u,r", [(20, 2, 4), (12, 3, 3), (30, 5, 2), (8, 1, 3)])
def test_eq10_p_miss_matches_combinations(n_pool, u, r):
    got = miss_probability(n_pool, u, r)
    exact = math.comb(n_pool - u, r) / math.comb(n_pool, r)
    assert got == exact
    bound = miss_probability_bound(n_pool, u, r)
    assert got <= bound + 1e-15


def test_eq10_edge_cases():
    assert miss_probability(10, 3, 8) == 0.0
    assert miss_probability(10, 0, 4) == 1.0
    assert miss_probability(10, 2, 0) == 1.0
    with pytest.raises(ValueError):
        miss_probability(5, 1, 6)


def test_hysteresis_holds_a_just_refreshed_row():
    n_q, n_k, k, m, r = 1, 10, 2, 1, 0
    scores = _ranked_scores(n_q, n_k)
    key = jax.random.PRNGKey(2)
    ledger = build_ledger(scores, k, m, r, key)
    assert int(ledger.ages[0]) == 0
    scores2 = scores.at[0, 2].set(50.0)
    k_c = jnp.eye(n_k, dtype=scores.dtype)
    held, info = relay_layer(
        scores2, k_c, ledger, k=k, m=m, r=r, gamma=0.0, rng_key=key
    )
    assert bool(info.reuse[0])
    assert jnp.all(held.routes == ledger.routes)
    refreshed, info2 = relay_layer(
        scores2, k_c, held, k=k, m=m, r=r, gamma=0.0, rng_key=key
    )
    assert not bool(info2.reuse[0])
    assert int(refreshed.ages[0]) == 0
    exact = jnp.sort(topk_routes(scores2, k), axis=1)
    assert jnp.all(jnp.sort(refreshed.routes, axis=1) == exact)
