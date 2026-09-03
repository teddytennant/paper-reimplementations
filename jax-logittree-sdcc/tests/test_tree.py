"""The bookkeeping has to reproduce the paper's running example exactly."""

import pytest

from memorywalker.tree import Eviction, Rollout, running_example

from conftest import random_rollout


def test_counts_match_definition_2():
    r = running_example()
    assert r.n_junctions == 3
    assert len(r.branches()) == 4  # K+1 root-to-leaf branches


def test_branches_match_figure_2a():
    r = running_example()
    assert r.branch(0) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert r.branch(1) == (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
    assert r.branch(2) == (0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12)
    assert r.branch(3) == (0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13)
    assert r.branch(3) == r.compressed_walk()  # the deployed walk is the last leg


def test_layers_partition_the_response():
    r = running_example()
    assert r.layer_targets(0) == (1, 2, 3, 4, 5, 6, 7)
    assert r.layer_targets(1) == (8, 9, 10)
    assert r.layer_targets(2) == (11, 12)
    assert r.layer_targets(3) == (13,)
    seen = [t for k in range(4) for t in r.layer_targets(k)]
    assert sorted(seen) == list(r.response_positions())
    assert len(seen) == len(set(seen))  # no leaf counted twice


def test_table_4_live_and_final_prefixes():
    """Table 4, App. B.4, verbatim. Physical index i is the paper's y_i."""
    r = running_example()
    expected = {
        6: ((1, 2, 3, 4, 5), (1, 2, 3, 4), {5}),
        7: ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 6), {5}),
        9: ((1, 2, 3, 4, 6, 7, 8), (1, 2, 3, 4, 6, 7), {8}),
        10: ((1, 2, 3, 4, 6, 7, 8, 9), (1, 2, 3, 4, 6, 7, 9), {8}),
        12: ((1, 2, 3, 4, 6, 7, 9, 10, 11), (1, 2, 3, 4, 6, 7, 9, 10), {11}),
    }
    for t, (live, final, diff) in expected.items():
        assert r.live_view(t)[1:] == live       # index 0 is the system prompt
        assert r.compressed_prefix(t)[1:] == final
        assert set(r.live_view(t)) - set(r.compressed_prefix(t)) == diff
    # y_ans re-converges: the tree needs no correction there.
    assert r.live_view(13) == r.compressed_prefix(13)


def test_diverging_leaf_set_matches_table_4():
    assert running_example().diverging_leaves() == (6, 7, 9, 10, 12)


def test_no_evictions_is_a_single_branch():
    r = Rollout(tokens=tuple(range(8)))
    assert r.n_junctions == 0
    assert r.branches() == (tuple(range(8)),)
    assert r.diverging_leaves() == ()
    assert r.compressed_walk() == r.physical_trace()


@pytest.mark.parametrize("seed", range(6))
def test_random_schedules_keep_the_invariants(seed):
    r = random_rollout(seed)
    branches = r.branches()
    assert len(branches) == r.n_junctions + 1
    covered = []
    for k, seq in enumerate(branches):
        for t in r.layer_targets(k):
            assert t in seq
            # branch k below t is exactly the live view of t
            assert tuple(i for i in seq if i < t) == r.live_view(t)
            covered.append(t)
    assert sorted(covered) == list(r.response_positions())
    for t in r.diverging_leaves():
        assert r.eviction_mass(t) > 0


def test_rejects_impossible_records():
    with pytest.raises(ValueError):
        Eviction(junction=3, span=(5,))
    with pytest.raises(ValueError):
        Rollout(tokens=tuple(range(6)),
                evictions=(Eviction(4, (2,)), Eviction(3, (1,))))
    with pytest.raises(ValueError):
        Rollout(tokens=tuple(range(6)),
                evictions=(Eviction(3, (2,)), Eviction(4, (2,))))
