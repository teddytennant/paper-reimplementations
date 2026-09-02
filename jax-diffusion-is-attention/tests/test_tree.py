"""Sec. IV: does the tree find the neighbours, and does it cost less than O(|D|)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffusion_is_attention import (
    SimilarityTree,
    attention_weights,
    exact_topk,
    optimal_denoiser,
    tree_attention,
)


def clustered(key, n_clusters, per_cluster, dim=8, spread=0.05, radius=6.0):
    kc, kp = jax.random.split(key)
    centers = jax.random.normal(kc, (n_clusters, dim)) * radius
    pts = centers[:, None, :] + spread * jax.random.normal(
        kp, (n_clusters, per_cluster, dim)
    )
    return pts.reshape(-1, dim), centers


def queries_near_clusters(key, centers, spread=0.05, n=32):
    kq, kn = jax.random.split(key)
    idx = jax.random.randint(kq, (n,), 0, centers.shape[0])
    return centers[idx] + spread * jax.random.normal(kn, (n, centers.shape[-1]))


def recall_at(data, centers, k, max_leaves, leaf_size=32, n_queries=32):
    tree = SimilarityTree(data, branching=4, leaf_size=leaf_size)
    qs = queries_near_clusters(jax.random.key(1), centers, n=n_queries)
    hits = 0
    for q in qs:
        got = set(tree.query(q, k=k, max_leaves=max_leaves).indices.tolist())
        got &= set(exact_topk(data, q, k).indices.tolist())
        hits += len(got)
    return hits / (k * n_queries)


def test_recall_against_brute_force():
    """Cheap queries already recover most of the true top-8, and 8 leaves get all.

    The misses at max_leaves=2 are queries whose cluster sits just across a
    Voronoi boundary from the branch representative it was assigned to. That is
    what the best-first backtracking is for, and it costs more leaf scans.
    """
    data, centers = clustered(jax.random.key(0), 64, 32)
    assert recall_at(data, centers, k=8, max_leaves=2) > 0.9
    assert recall_at(data, centers, k=8, max_leaves=8) == 1.0


def test_recall_improves_with_more_leaves():
    """When a cluster spans several leaves, one leaf is not enough to see all of it."""
    data, centers = clustered(jax.random.key(0), 32, 128)
    curve = [recall_at(data, centers, k=8, max_leaves=m) for m in (1, 2, 4, 8, 16)]
    assert curve[0] < curve[-1]
    assert all(a <= b + 1e-12 for a, b in zip(curve, curve[1:]))
    assert curve[-1] > 0.95


def test_tree_attention_matches_attention_over_the_exact_neighbours():
    """With full recall the tree changes nothing about the attention output."""
    data, centers = clustered(jax.random.key(2), 32, 32)
    tree = SimilarityTree(data, branching=4, leaf_size=32)
    for q in queries_near_clusters(jax.random.key(3), centers, n=16):
        approx, hit = tree_attention(tree, q, 0.1, k=8, max_leaves=8)
        want_idx = exact_topk(data, q, 8).indices
        assert sorted(hit.indices) == sorted(want_idx)
        exact = optimal_denoiser(q, data[np.sort(want_idx)], 0.1)
        assert float(jnp.linalg.norm(approx - exact)) < 1e-11


def test_restricting_to_neighbours_is_accurate_only_at_small_sigma():
    """Sec. IV leans on sigma being small. Here is where that stops being true."""
    data, centers = clustered(jax.random.key(4), 32, 32)
    tree = SimilarityTree(data, branching=4, leaf_size=32)
    q = queries_near_clusters(jax.random.key(5), centers, n=1)[0]

    tight, _ = tree_attention(tree, q, 0.1, k=32, max_leaves=8)
    assert float(jnp.linalg.norm(tight - optimal_denoiser(q, data, 0.1))) < 1e-9

    # At sigma = 20 the far clusters carry most of the weight and truncation shows.
    loose, hit = tree_attention(tree, q, 20.0, k=32, max_leaves=8)
    kept = float(jnp.sum(attention_weights(q, data, 20.0)[hit.indices]))
    assert kept < 0.1
    assert float(jnp.linalg.norm(loose - optimal_denoiser(q, data, 20.0))) > 1.0


def test_query_cost_is_sublinear_in_dataset_size():
    """Cost tracks the tree depth, not |D|."""
    costs = {}
    for n_clusters in (16, 64, 256):
        data, centers = clustered(jax.random.key(6), n_clusters, 32)
        tree = SimilarityTree(data, branching=4, leaf_size=32)
        qs = queries_near_clusters(jax.random.key(7), centers, n=8)
        evals = [tree.query(q, k=8, max_leaves=2).distance_evals for q in qs]
        costs[data.shape[0]] = float(np.mean(evals))

    small, large = min(costs), max(costs)
    # At 512 points a leaf scan is already an eighth of the data, so the
    # interesting bound is at the top of the range and in how the cost scales.
    assert costs[large] < 0.02 * large
    size_ratio = large / small
    cost_ratio = costs[large] / costs[small]
    assert cost_ratio < 0.25 * size_ratio
    assert cost_ratio < np.sqrt(size_ratio)


def test_tree_depth_is_logarithmic():
    data, _ = clustered(jax.random.key(8), 128, 32)
    tree = SimilarityTree(data, branching=4, leaf_size=32)
    assert tree.depth <= 3 * int(np.ceil(np.log(data.shape[0]) / np.log(4)))


def test_every_point_is_reachable():
    data, _ = clustered(jax.random.key(9), 8, 16)
    tree = SimilarityTree(data, branching=3, leaf_size=8)
    seen = set()

    def walk(node):
        if node.is_leaf:
            seen.update(int(m) for m in node.members)
        for c in node.children:
            walk(c)

    walk(tree.root)
    assert seen == set(range(data.shape[0]))


def test_exact_query_on_a_tiny_dataset():
    data = jax.random.normal(jax.random.key(10), (40, 3))
    tree = SimilarityTree(data, branching=2, leaf_size=40)
    q = jax.random.normal(jax.random.key(11), (3,))
    got = tree.query(q, k=5)
    want = exact_topk(data, q, 5)
    assert list(got.indices) == list(want.indices)
    assert got.distance_evals == data.shape[0]


def test_query_raises_when_it_cannot_gather_k_candidates():
    data = jax.random.normal(jax.random.key(12), (256, 4))
    tree = SimilarityTree(data, branching=4, leaf_size=4)
    with pytest.raises(ValueError):
        tree.query(data[0], k=200, max_leaves=1)
