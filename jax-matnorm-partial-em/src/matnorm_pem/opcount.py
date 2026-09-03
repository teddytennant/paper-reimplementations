"""A counter for the matrix factorizations each E-step performs.

The paper's efficiency argument is about pattern-specific inversions: the exact
E-step of (2), or its precision form (3), factorizes one matrix per missingness
pattern per iteration whose size grows with the number of missing cells, while
the partial E-step of (6), (8) and (9) factorizes nothing beyond the p x p and
q x q covariance factors that the M-step needs anyway.

Wall-clock is noisy, so the tests count these events instead. Recording happens
in the Python drivers, outside jit, so the counts are exact and do not depend on
what XLA fuses.
"""

from typing import List, NamedTuple


class Event(NamedTuple):
    kind: str  # "estep" or "mstep"
    dim: int  # order of the matrix being factorized
    count: int  # how many of them


_EVENTS: List[Event] = []
_ENABLED = False


def reset(enabled: bool = True) -> None:
    global _ENABLED
    _EVENTS.clear()
    _ENABLED = enabled


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def record(kind: str, dim: int, count: int = 1) -> None:
    if _ENABLED and count > 0 and dim > 0:
        _EVENTS.append(Event(kind, int(dim), int(count)))


def events() -> List[Event]:
    return list(_EVENTS)


def total(kind: str = None) -> int:
    """Number of factorizations recorded, optionally restricted to one kind."""
    return sum(e.count for e in _EVENTS if kind is None or e.kind == kind)


def max_dim(kind: str = None) -> int:
    """Largest matrix order factorized, 0 if nothing was recorded."""
    dims = [e.dim for e in _EVENTS if kind is None or e.kind == kind]
    return max(dims) if dims else 0


def work(kind: str = None) -> int:
    """Sum of dim^3 over recorded factorizations, a flop-order summary."""
    return sum(e.count * e.dim**3 for e in _EVENTS if kind is None or e.kind == kind)
