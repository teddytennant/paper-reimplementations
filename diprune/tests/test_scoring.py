"""Equation checks for DIPrune visual-token scores."""

import jax
import numpy as np
import pytest

from diprune import (
    aggregate_attention,
    alignment_score,
    decoupled_attention,
    fuse_scores,
    ranks_from_scores,
    semantic_alignment,
    semantic_evolution,
    window_length,
)

jax.config.update("jax_enable_x64", True)


def test_unique_extrema_and_positive_scale_invariance():
    scores = np.array([0.2, 1.5, 0.05, 0.9, 0.4])
    n = scores.shape[0]
    got = semantic_alignment(scores)
    assert got[int(np.argmax(scores))] == 1.0
    assert got[int(np.argmin(scores))] == 1.0 / n
    scaled = semantic_alignment(scores * 3.7)
    np.testing.assert_allclose(got, scaled)


def test_rank_ties_give_the_smaller_index_the_better_rank():
    scores = np.array([0.2, 0.5, 0.5, 0.1])
    ranks = ranks_from_scores(scores)
    assert ranks.tolist() == [3, 1, 2, 4]
    aligned = alignment_score(ranks)
    n = scores.shape[0]
    expected = (n - np.array([3, 1, 2, 4]) + 1) / n
    np.testing.assert_allclose(aligned, expected)


def test_evolution_midpoint_and_clip():
    m_max = 200
    ranks_prev = np.array([0, m_max, -m_max, 2 * m_max])
    ranks_curr = np.array([0, 0, 0, 0])
    got = semantic_evolution(ranks_prev, ranks_curr, m_max=m_max)
    assert got[0] == 0.5
    assert got[1] == 1.0
    assert got[2] == 0.0
    assert got[3] == got[1]


def test_fusion_is_the_convex_combination():
    s_align = np.array([1.0, 0.5, 0.0])
    s_evol = np.array([0.0, 0.25, 1.0])
    got = fuse_scores(s_align, s_evol, alpha=0.6, beta=0.4)
    expected = 0.6 * s_align + 0.4 * s_evol
    np.testing.assert_allclose(got, expected)
    np.testing.assert_allclose(fuse_scores(s_align, s_evol), expected)
    with pytest.raises(ValueError):
        fuse_scores(s_align, s_evol, alpha=0.5, beta=0.2)


def test_alignment_and_evolution_disagree_on_top1():
    # Order: static, rising, mid, low, drop.
    ranks_prev = np.array([1, 5, 2, 3, 4])
    ranks_curr = np.array([1, 2, 3, 4, 5])
    m_max = 2
    n = ranks_curr.shape[0]
    s_align_expected = (n - ranks_curr + 1) / n
    delta = ranks_prev - ranks_curr
    clipped = np.clip(delta, -m_max, m_max)
    s_evol_expected = 0.5 * (clipped / m_max + 1.0)
    np.testing.assert_allclose(s_align_expected, [1.0, 0.8, 0.6, 0.4, 0.2])
    np.testing.assert_allclose(s_evol_expected, [0.5, 1.0, 0.25, 0.25, 0.25])

    s_align = alignment_score(ranks_curr)
    s_evol = semantic_evolution(ranks_prev, ranks_curr, m_max=m_max)
    np.testing.assert_allclose(s_align, s_align_expected)
    np.testing.assert_allclose(s_evol, s_evol_expected)

    only_align = np.asarray(fuse_scores(s_align, s_evol, alpha=1.0, beta=0.0))
    only_evol = np.asarray(fuse_scores(s_align, s_evol, alpha=0.0, beta=1.0))
    assert int(np.argmax(only_align)) == 0
    assert int(np.argmax(only_evol)) == 1
    assert int(np.argmax(s_align_expected)) == 0
    assert int(np.argmax(s_evol_expected)) == 1


def _numpy_window_saliency(queries, visual_keys, n_heads, window_ratio):
    """Independent softmax reference. Heads are contiguous blocks."""
    queries = np.asarray(queries, dtype=np.float64)
    visual_keys = np.asarray(visual_keys, dtype=np.float64)
    n_queries, dim = queries.shape
    length = max(1, min(n_queries, round(window_ratio * n_queries)))
    q_sub = queries[-length:]
    d_h = dim // n_heads
    scale = np.sqrt(np.float64(d_h))
    heads = []
    for head in range(n_heads):
        start = head * d_h
        stop = start + d_h
        logits = (q_sub[:, start:stop] @ visual_keys[:, start:stop].T) / scale
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        heads.append(exp_logits / exp_logits.sum(axis=1, keepdims=True))
    averaged = np.mean(heads, axis=0)
    return averaged.mean(axis=0)


def test_decoupled_attention_matches_numpy_softmax():
    queries = np.array(
        [
            [1.0, 0.0, 0.5, -0.5],
            [0.0, 1.0, -1.0, 0.5],
        ]
    )
    visual_keys = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    got = decoupled_attention(queries, visual_keys, n_heads=2, window_ratio=1.0)
    expected = _numpy_window_saliency(queries, visual_keys, n_heads=2, window_ratio=1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert got.shape == (3,)


def test_half_window_ignores_queries_outside_the_last_rows():
    assert window_length(4, 0.5) == 2
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(4, 8))
    visual_keys = rng.normal(size=(5, 8))
    base = decoupled_attention(queries, visual_keys, n_heads=2, window_ratio=0.5)
    outside = queries.copy()
    outside[0] += 10.0
    unchanged = decoupled_attention(outside, visual_keys, n_heads=2, window_ratio=0.5)
    np.testing.assert_allclose(base, unchanged)
    also_outside = queries.copy()
    also_outside[1] += 10.0
    still = decoupled_attention(also_outside, visual_keys, n_heads=2, window_ratio=0.5)
    np.testing.assert_allclose(base, still)
    inside = queries.copy()
    inside[-2] += 10.0
    changed = decoupled_attention(inside, visual_keys, n_heads=2, window_ratio=0.5)
    assert not np.allclose(base, changed)


def test_decoupled_attention_rejects_indivisible_width():
    with pytest.raises(ValueError):
        decoupled_attention(np.ones((2, 5)), np.ones((3, 5)), n_heads=2)


def test_aggregate_attention_means_queries_and_heads():
    attention = np.array(
        [
            [0.2, 0.8, 0.0],
            [0.5, 0.1, 0.4],
            [0.0, 0.0, 1.0],
            [0.3, 0.3, 0.4],
        ]
    )
    expected = attention.sum(axis=0) / attention.shape[0]
    np.testing.assert_allclose(aggregate_attention(attention), expected)

    heads = np.array(
        [
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.2]],
            [[0.0, 1.0, 0.5], [1.0, 0.0, 0.8]],
            [[0.5, 0.5, 0.0], [0.2, 0.3, 0.1]],
        ]
    )
    headed = heads.mean(axis=0)
    expected_3d = headed.sum(axis=0) / headed.shape[0]
    np.testing.assert_allclose(aggregate_attention(heads), expected_3d)
    np.testing.assert_allclose(aggregate_attention(heads), heads.mean(axis=(0, 1)))
