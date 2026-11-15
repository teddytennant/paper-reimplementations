"""Cluster-then-Select, lines 12 to 16 of Algorithm 6."""

import jax
import numpy as np
import pytest

from ugtl.generation import cluster_then_select
from ugtl.kmeans import kmeans


def spread(points: np.ndarray) -> float:
    """Mean pairwise distance, the thing clustering is supposed to buy."""
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(d.sum() / (len(points) * (len(points) - 1)))


def test_returns_exactly_q_distinct_designs():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(600, 4))
    proxy = rng.normal(size=600)
    for Q in (2, 5, 16):
        idx = cluster_then_select(jax.random.PRNGKey(0), pool, proxy, Q, 200)
        assert len(idx) == Q
        assert len(set(idx.tolist())) == Q
        assert np.unique(pool[idx], axis=0).shape[0] == Q


def test_each_pick_is_the_proxy_argmax_of_its_cluster():
    rng = np.random.default_rng(1)
    pool = rng.normal(size=(400, 3))
    proxy = rng.normal(size=400)
    Q, M_pool = 6, 150
    key = jax.random.PRNGKey(3)
    idx = cluster_then_select(key, pool, proxy, Q, M_pool)

    keep = np.argsort(-proxy, kind="stable")[:M_pool]
    top = pool[keep]
    assign, _ = kmeans(key, (top - top.mean(0)) / (top.std(0) + 1e-6), Q)
    assign = np.asarray(assign)
    for j in idx:
        members = keep[assign == assign[np.flatnonzero(keep == j)[0]]]
        assert proxy[j] == proxy[members].max()


def test_only_the_top_m_pool_designs_are_eligible():
    rng = np.random.default_rng(2)
    pool = rng.normal(size=(300, 2))
    proxy = rng.normal(size=300)
    M_pool = 40
    idx = cluster_then_select(jax.random.PRNGKey(0), pool, proxy, 5, M_pool)
    assert set(idx.tolist()) <= set(np.argsort(-proxy, kind="stable")[:M_pool].tolist())


def test_it_spreads_the_budget_when_the_top_scores_sit_in_one_mode():
    """The pool has one tight high-proxy mode and several lower ones. Plain top-Q
    spends the whole budget inside the tight mode; Cluster-then-Select does not.
    """
    rng = np.random.default_rng(3)
    modes = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0], [4.0, -6.0]])
    pool, proxy = [], []
    for m, centre in enumerate(modes):
        pts = centre + rng.normal(scale=0.15, size=(120, 2))
        pool.append(pts)
        # The first mode holds every one of the top scores.
        proxy.append(rng.uniform(2.0, 3.0, 120) if m == 0 else rng.uniform(0.0, 1.0, 120))
    pool = np.concatenate(pool)
    proxy = np.concatenate(proxy)

    Q = 5
    top_q = np.argsort(-proxy, kind="stable")[:Q]
    picked = cluster_then_select(jax.random.PRNGKey(0), pool, proxy, Q, 300)

    assert spread(pool[picked]) > 5.0 * spread(pool[top_q])
    assert len({int(np.argmin(np.linalg.norm(modes - p, axis=1))) for p in pool[picked]}) > 1
    assert len({int(np.argmin(np.linalg.norm(modes - p, axis=1))) for p in pool[top_q]}) == 1


def test_it_still_takes_the_single_best_design():
    """Spreading the budget must not throw away the top proxy score."""
    rng = np.random.default_rng(4)
    pool = rng.normal(size=(300, 3))
    proxy = rng.normal(size=300)
    idx = cluster_then_select(jax.random.PRNGKey(1), pool, proxy, 8, 200)
    assert int(np.argmax(proxy)) in idx.tolist()


def test_budget_larger_than_the_pool_is_rejected():
    with pytest.raises(ValueError):
        cluster_then_select(jax.random.PRNGKey(0), np.zeros((3, 2)), np.zeros(3), 5, 10)
