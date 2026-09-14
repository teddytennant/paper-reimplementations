"""Prüfer encode/decode (arXiv:2609.09255 Appendix A).

Vertices may use any totally ordered labels. A code of length m-2 on a
label set of size m is a bijection with labeled spanning trees. Encoding
repeatedly deletes the smallest-labeled leaf and records its neighbor.
Decoding follows the leaf-pointer algorithm in the appendix: initialize
r_p = 1 + multiplicity of p in the code; at each step connect the
smallest remaining label of residual 1 to the next code symbol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def prufer_encode(
    edges: Iterable[tuple[int, int]],
    vertices: Sequence[int],
) -> list[int]:
    """Encode a labeled tree as a Prüfer sequence of length m-2."""
    vertices = list(vertices)
    m = len(vertices)
    if m <= 2:
        return []

    adj: dict[int, set[int]] = {v: set() for v in vertices}
    for u, w in edges:
        adj[u].add(w)
        adj[w].add(u)

    remaining = set(vertices)
    code: list[int] = []
    for _ in range(m - 2):
        leaf = min(v for v in remaining if len(adj[v]) == 1)
        nbr = next(iter(adj[leaf]))
        code.append(nbr)
        adj[nbr].remove(leaf)
        adj[leaf].clear()
        remaining.remove(leaf)
    return code


def prufer_decode(
    code: Sequence[int],
    vertices: Sequence[int],
) -> list[tuple[int, int]]:
    """Decode a Prüfer sequence into undirected tree edges.

    When m = 2 the code is empty and the tree is the single edge between
    the two labels.
    """
    vertices = list(vertices)
    m = len(vertices)
    if m < 2:
        return []
    if m == 2:
        u, w = vertices[0], vertices[1]
        return [(u, w)] if u <= w else [(w, u)]

    code = list(code)
    if len(code) != m - 2:
        raise ValueError(f"Prüfer code length {len(code)} != m-2 = {m - 2}")

    residual = {v: 1 for v in vertices}
    for symbol in code:
        residual[symbol] += 1

    remaining = set(vertices)
    edges: list[tuple[int, int]] = []
    for symbol in code:
        leaf = min(v for v in remaining if residual[v] == 1)
        a, b = (leaf, symbol) if leaf <= symbol else (symbol, leaf)
        edges.append((a, b))
        remaining.remove(leaf)
        residual[leaf] = 0
        residual[symbol] -= 1

    rest = sorted(remaining)
    if len(rest) != 2:
        raise ValueError(f"expected two labels at the end, got {rest}")
    edges.append((rest[0], rest[1]))
    return edges
