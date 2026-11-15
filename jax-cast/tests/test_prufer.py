"""Prüfer correspondence tests (paper Appendix A example)."""

from __future__ import annotations

from jax_cast.prufer import prufer_decode, prufer_encode


def _undirected(edges):
    return {frozenset(e) for e in edges}


def test_prufer_paper_example_decode():
    vertices = [1, 2, 3, 4, 5]
    edges = prufer_decode((3, 3, 4), vertices)
    expected = {(1, 3), (2, 3), (3, 4), (4, 5)}
    assert _undirected(edges) == {frozenset(e) for e in expected}


def test_prufer_paper_example_encode():
    vertices = [1, 2, 3, 4, 5]
    tree = [(1, 3), (2, 3), (3, 4), (4, 5)]
    assert prufer_encode(tree, vertices) == [3, 3, 4]


def test_prufer_m2_empty_code():
    edges = prufer_decode((), [0, 1])
    assert _undirected(edges) == {frozenset((0, 1))}
    assert prufer_encode(edges, [0, 1]) == []


def test_prufer_roundtrip_zero_based():
    vertices = list(range(6))
    code = [2, 0, 4, 2]
    edges = prufer_decode(code, vertices)
    assert prufer_encode(edges, vertices) == code
