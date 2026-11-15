"""Eviction bookkeeping for a compressed rollout.

Follows Definition 2 (App. B.1) of arXiv:2609.00865. A rollout is a physical
token sequence ``H_T = (x_0, ..., x_{T-1})`` plus eviction records
``{(J_k, E_k)}_{k=1..K}`` with ``J_1 < ... < J_K``: the k-th edit fires at
physical position ``J_k`` and removes the physical positions in ``E_k``.

Timing convention. The paper writes the live view of Eq. 2 as "the physical
history with exactly the edits that have fired by step t applied, and no
others", and separately says the k-th branch layer is the set of positions in
``[J_{k-1}, J_k)``. Those two statements are off by one from each other. We
follow Table 4 and Figure 2, which are unambiguous: with ``E_1 = {y_5}`` at
``J_1 = 7``, the target ``y_7`` still conditions on ``y_5`` while ``y_9`` does
not. So an edit at ``J_k`` applies to decode positions ``t > J_k``, and branch
layer k owns the targets ``J_k < t <= J_{k+1}``. This is the reading that makes
Table 4's diverging-leaf set come out exactly right, and it is the one under
which the gradient-equivalence claim of Theorem 1 holds.

Positions are 0-indexed here; the paper's running example is 1-indexed with the
system prompt at index 0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Eviction:
    """One eviction record ``(J_k, E_k)``.

    ``junction`` is the physical position at which the edit fires; ``span`` is
    the set of physical positions it removes, a subset of ``{0, ..., J_k}``.
    """

    junction: int
    span: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(p > self.junction for p in self.span):
            raise ValueError("an eviction cannot remove a position it has not reached yet")
        if len(set(self.span)) != len(self.span):
            raise ValueError("duplicate position in an eviction span")


@dataclass(frozen=True)
class Rollout:
    """A compressed rollout: physical tokens, eviction records, prompt length.

    ``response_start`` is the first loss-carrying position. Positions before it
    are the prompt and are never scored.
    """

    tokens: tuple[int, ...]
    evictions: tuple[Eviction, ...] = ()
    response_start: int = 1

    def __post_init__(self) -> None:
        js = [e.junction for e in self.evictions]
        if js != sorted(js) or len(set(js)) != len(js):
            raise ValueError("eviction junctions must be strictly increasing")
        seen: set[int] = set()
        for e in self.evictions:
            if seen & set(e.span):
                raise ValueError("a position cannot be evicted twice")
            seen |= set(e.span)
        if any(p < self.response_start for e in self.evictions for p in e.span):
            raise ValueError("the prompt cannot be evicted")

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def n_junctions(self) -> int:
        """K, the junction depth of the trajectory tree."""
        return len(self.evictions)

    def evicted_before(self, t: int) -> set[int]:
        """Positions removed by the edits that had fired when ``x_t`` was decoded."""
        out: set[int] = set()
        for e in self.evictions:
            if e.junction < t:
                out |= set(e.span)
        return out

    @property
    def all_evicted(self) -> set[int]:
        return {p for e in self.evictions for p in e.span}

    def live_view(self, t: int) -> tuple[int, ...]:
        """``p_t(y_t) = H'_t`` of Eq. 2: the prefix visible when ``x_t`` was decoded."""
        gone = self.evicted_before(t)
        return tuple(i for i in range(t) if i not in gone)

    def compressed_prefix(self, t: int) -> tuple[int, ...]:
        """``p_s(y_t) = H_comp[:t]`` of Eq. 3: the prefix of the final walk."""
        gone = self.all_evicted
        return tuple(i for i in range(t) if i not in gone)

    def compressed_walk(self) -> tuple[int, ...]:
        """``H_comp``, the compressed spine that the harness ships at deployment."""
        gone = self.all_evicted
        return tuple(i for i in range(self.length) if i not in gone)

    def physical_trace(self) -> tuple[int, ...]:
        """``H_T``, the depth-first traversal that keeps every generated token."""
        return tuple(range(self.length))

    def response_positions(self) -> tuple[int, ...]:
        return tuple(range(self.response_start, self.length))

    def branch(self, k: int) -> tuple[int, ...]:
        """Physical positions on root-to-leaf branch ``k``, for ``k`` in ``0..K``.

        Branch k carries the live view in force during layer k, extended through
        the last target that layer owns. Branch K is ``H_comp``.
        """
        if not 0 <= k <= self.n_junctions:
            raise IndexError(f"branch {k} out of range for K={self.n_junctions}")
        gone: set[int] = set()
        for e in self.evictions[:k]:
            gone |= set(e.span)
        end = self.length if k == self.n_junctions else self.evictions[k].junction + 1
        return tuple(i for i in range(end) if i not in gone)

    def branches(self) -> tuple[tuple[int, ...], ...]:
        """The ``K+1`` root-to-leaf branches, ``RootToLeaf(T)`` of Eq. 6."""
        return tuple(self.branch(k) for k in range(self.n_junctions + 1))

    def layer_of(self, t: int) -> int:
        """The branch layer that owns target ``t``: the unique k with ``J_k < t <= J_{k+1}``."""
        k = 0
        for e in self.evictions:
            if e.junction < t:
                k += 1
        return k

    def layer_targets(self, k: int) -> tuple[int, ...]:
        """Loss-carrying positions scored on branch ``k``.

        The K+1 layers partition the response, which is what makes the per-branch
        loss masks "disjoint and jointly exhaustive" in Proposition 3.
        """
        return tuple(t for t in self.response_positions() if self.layer_of(t) == k)

    def diverging_leaves(self) -> tuple[int, ...]:
        """``C`` of Eq. 8: response tokens whose two prefixes disagree.

        A leaf diverges iff some edit that fires at or after t reaches back into
        its prefix (App. B.1). Evicted positions are not student leaves, since
        they are absent from ``H_comp`` entirely.
        """
        alive = set(self.compressed_walk())
        out = []
        for t in self.response_positions():
            if t not in alive:
                continue
            if set(self.live_view(t)) - set(self.compressed_prefix(t)):
                out.append(t)
        return tuple(out)

    def eviction_mass(self, t: int) -> int:
        """``|phi_leg \\ phi_spine|`` at ``t``: tokens the final walk no longer has."""
        return len(set(self.live_view(t)) - set(self.compressed_prefix(t)))

    def active_evictions(self, t: int) -> tuple[Eviction, ...]:
        """``A(p)`` of App. D.1: the evictions reinserted by SDCC's teacher at ``t``."""
        return tuple(
            e
            for e in self.evictions
            if e.junction >= t and any(p < t for p in e.span)
        )


def running_example() -> Rollout:
    """Figure 1's example: ``E_{1,2,3} = {y_5}, {y_8}, {y_11}`` at ``J = 7, 10, 12``.

    Fourteen physical positions: the system prompt, ``y_1..y_12`` and ``y_ans``.
    Token ids are arbitrary, only the structure is the paper's.
    """
    return Rollout(
        tokens=tuple(range(14)),
        evictions=(
            Eviction(junction=7, span=(5,)),
            Eviction(junction=10, span=(8,)),
            Eviction(junction=12, span=(11,)),
        ),
        response_start=1,
    )
