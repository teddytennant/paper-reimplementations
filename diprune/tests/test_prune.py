"""Checks that pruning keeps text tokens and the selected visual tokens."""

import numpy as np
import pytest

from diprune import prune_tokens, select_topk


def test_keeps_selected_visual_tokens_and_trailing_text():
    tokens = np.array([10, 11, 12, 13, 20, 21])
    visual_mask = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    pruned, kept = prune_tokens(tokens, visual_mask, scores, k=2)
    assert pruned.shape[0] == 4
    assert kept.tolist() == [0, 2, 4, 5]
    assert pruned.tolist() == [10, 12, 20, 21]
    assert pruned[-2:].tolist() == [20, 21]


def test_text_tokens_stay_when_only_one_visual_token_is_kept():
    tokens = np.array([10, 11, 12, 13, 20, 21])
    visual_mask = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.2, 0.9, 0.1, 0.0])
    pruned, kept = prune_tokens(tokens, visual_mask, scores, k=1)
    assert pruned.shape[0] == 3
    assert kept.tolist() == [1, 4, 5]
    assert pruned.tolist() == [11, 20, 21]


def test_k_out_of_range_raises():
    tokens = np.arange(6)
    visual_mask = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    with pytest.raises(ValueError):
        prune_tokens(tokens, visual_mask, scores, k=0)
    with pytest.raises(ValueError):
        prune_tokens(tokens, visual_mask, scores, k=5)
    with pytest.raises(ValueError):
        prune_tokens(tokens, visual_mask, scores, k=-1)


def test_selection_tie_prefers_the_smaller_index_and_keeps_order():
    scores = np.array([0.4, 0.9, 0.9, 0.1])
    assert select_topk(scores, 1).tolist() == [1]
    assert select_topk(scores, 2).tolist() == [1, 2]
    tokens = np.array([0, 10, 1, 11, 2])
    visual_mask = np.array([True, False, True, False, True])
    pruned, kept = prune_tokens(tokens, visual_mask, np.array([0.1, 0.9, 0.2]), k=1)
    assert kept.tolist() == [1, 2, 3]
    assert pruned.tolist() == [10, 1, 11]
