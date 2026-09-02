"""Sec. IV, tree cross attention.

Eq (11) needs the distance from the query to every image in D. Sec. IV says that
since sigma is small only the near signals matter, so if the dataset is stored in
a similarity tree the query can find them with about O(log |D|) comparisons
instead of O(|D|).

The paper describes building the tree by online insertion: test a new image
against the nodes at each level, descend the similar branch or open a new one.
That puts no bound on the branching factor, and the O(log |D|) claim needs one,
so this builds the same shape top down. Split a node's points around `branching`
representatives picked by farthest-point seeding, recurse until a node holds at
most `leaf_size` points. Every node is still an image from the dataset, as in the
paper. Queries walk best-first so a point sitting between two branches is not
lost, and each query reports how many dataset distances it touched.

The tree is a host-side structure. It uses numpy internally and hands back jax
arrays, so it is not something to call inside a jit.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from .closed_form import _softmax

__all__ = ["SimilarityTree", "TopK", "exact_topk", "tree_attention"]


@dataclass
class _Node:
    rep: int
    members: np.ndarray
    children: list["_Node"] = field(default_factory=list)

    @property
    def is_leaf(self):
        return not self.children


@dataclass
class TopK:
    """Result of a query. `distance_evals` is the whole cost, in point distances."""

    indices: np.ndarray
    distances: np.ndarray
    distance_evals: int


def _sq(a, b):
    """Squared distances between rows of `a` and rows of `b`, shape (len(a), len(b))."""
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def _seed_centers(points, branching):
    """Farthest-point seeding, deterministic. Returns row positions in `points`.

    The first seed is the point farthest from the centroid, not the medoid.
    Seeding from the medoid leaves one enormous central Voronoi cell and the tree
    degenerates: on 8192 points in 8 dimensions it went 23 levels deep instead
    of 7, which would sink the O(log |D|) query cost Sec. IV is after.
    """
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(_sq(centroid, points)[0]))
    chosen = [first]
    best = _sq(points[first : first + 1], points)[0]
    for _ in range(branching - 1):
        nxt = int(np.argmax(best))
        if best[nxt] <= 0.0:
            break
        chosen.append(nxt)
        best = np.minimum(best, _sq(points[nxt : nxt + 1], points)[0])
    return chosen


class SimilarityTree:
    """A balanced similarity tree over a dataset, Sec. IV.

    >>> tree = SimilarityTree(data, branching=4, leaf_size=32)
    >>> hit = tree.query(q, k=8)
    """

    def __init__(self, data, branching=4, leaf_size=32):
        if branching < 2:
            raise ValueError("branching must be at least 2")
        self.data = jnp.asarray(data)
        self._points = np.asarray(self.data, dtype=np.float64)
        self.branching = int(branching)
        self.leaf_size = int(leaf_size)
        self.root = self._build(np.arange(self._points.shape[0]))

    def _build(self, members):
        if len(members) <= self.leaf_size:
            return _Node(rep=int(members[0]), members=members)
        positions = _seed_centers(self._points[members], self.branching)
        if len(positions) < 2:
            return _Node(rep=int(members[0]), members=members)
        centers = members[positions]
        assign = np.argmin(_sq(self._points[members], self._points[centers]), axis=1)
        node = _Node(rep=int(centers[0]), members=members)
        for j, center in enumerate(centers):
            group = members[assign == j]
            if len(group) == 0:
                continue
            child = self._build(group)
            child.rep = int(center)
            node.children.append(child)
        if len(node.children) < 2:
            node.children = []
        return node

    @property
    def depth(self):
        def go(node):
            return 1 if node.is_leaf else 1 + max(go(c) for c in node.children)

        return go(self.root)

    def query(self, q, k, max_leaves=2):
        """Approximate k nearest neighbours of `q`, and what the search cost.

        Best-first over branch representatives: expand the node whose
        representative is nearest, scan a leaf when one comes up, stop after
        `max_leaves` leaves. Representatives met on the way down are dataset
        points too, so they stay in the candidate pool rather than being thrown
        away. The reported cost counts every dataset point the query measured a
        distance to, including the final exact rescan of the candidates.
        """
        q = np.asarray(jnp.asarray(q), dtype=np.float64).reshape(1, -1)
        candidates: set[int] = set()
        evals = 0

        if self.root.is_leaf:
            candidates.update(int(m) for m in self.root.members)
            evals += len(self.root.members)
        else:
            heap: list[tuple[float, int, _Node]] = []
            tie = 0

            def push(nodes):
                nonlocal evals, tie
                reps = np.array([n.rep for n in nodes])
                d = _sq(q, self._points[reps])[0]
                evals += len(reps)
                candidates.update(int(r) for r in reps)
                for node, dist in zip(nodes, d):
                    heapq.heappush(heap, (float(dist), tie, node))
                    tie += 1

            push(self.root.children)
            leaves = 0
            while heap and leaves < max_leaves:
                _, _, node = heapq.heappop(heap)
                if node.is_leaf:
                    candidates.update(int(m) for m in node.members)
                    evals += len(node.members)
                    leaves += 1
                else:
                    push(node.children)

        idx = np.fromiter(sorted(candidates), dtype=np.int64, count=len(candidates))
        if len(idx) < k:
            raise ValueError(
                f"only {len(idx)} candidates for k={k}; raise max_leaves or leaf_size"
            )
        d = _sq(q, self._points[idx])[0]
        best = np.argsort(d, kind="stable")[:k]
        return TopK(indices=idx[best], distances=d[best], distance_evals=evals)


def exact_topk(data, q, k):
    """Brute-force top-K, the O(|D|) baseline Sec. IV is trying to avoid."""
    points = np.asarray(jnp.asarray(data), dtype=np.float64)
    d = _sq(np.asarray(jnp.asarray(q), dtype=np.float64).reshape(1, -1), points)[0]
    idx = np.argsort(d, kind="stable")[:k]
    return TopK(indices=idx, distances=d[idx], distance_evals=int(points.shape[0]))


def tree_attention(tree, q, sigma, k, max_leaves=2):
    """Eq (11) restricted to the neighbours the tree returns, Sec. IV.

    Sec. IV's argument for why this is close to the full sum: sigma is small, so
    the signals outside the retrieved set carry weights that are negligible next
    to the retrieved ones. That holds when the retrieved set really is the
    nearest one, which is what the recall test checks.
    """
    hit = tree.query(q, k=k, max_leaves=max_leaves)
    w = _softmax(-jnp.asarray(hit.distances) / (2.0 * jnp.asarray(sigma) ** 2))
    return w @ tree.data[hit.indices], hit
