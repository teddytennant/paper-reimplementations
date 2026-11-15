"""Lines of a vector allocation (Definition 9) and 1D binary search.

For x : [0:m]^n → [0:d]^n, the i-line with prefixes α = bids[:i] and
suffixes β = bids[i+1:] is

    x_{α,i,β}(b) = x(α, b, β)_i.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def assemble_bids(alpha: ArrayLike, b: int, beta: ArrayLike) -> NDArray[np.int_]:
    """Concatenate (α, b, β) into a length-n bid vector."""
    alpha_arr = np.asarray(alpha, dtype=int).reshape(-1)
    beta_arr = np.asarray(beta, dtype=int).reshape(-1)
    parts = []
    if alpha_arr.size:
        parts.append(alpha_arr)
    parts.append(np.array([int(b)], dtype=int))
    if beta_arr.size:
        parts.append(beta_arr)
    return np.concatenate(parts)


def eval_line(
    x: Callable[[NDArray[np.int_]], ArrayLike],
    alpha: ArrayLike,
    i: int,
    beta: ArrayLike,
    b: int,
):
    """Return x(α, b, β)_i."""
    bids = assemble_bids(alpha, b, beta)
    alloc = np.asarray(x(bids))
    return alloc[i]


def binary_search_first(
    x_fn: Callable[[int], int],
    m: int,
    ell: int,
) -> Optional[int]:
    """Smallest b in {0, ..., m} with x_fn(b) == ell, or None.

    Lowest-index binary search with lo=0, hi=m, comparing x(mid) to ell.
    If x is not monotone the search may miss; later checks catch that.
    """
    lo, hi = 0, m
    found: Optional[int] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        val = int(x_fn(mid))
        if val == ell:
            found = mid
            hi = mid - 1
        elif val < ell:
            lo = mid + 1
        else:
            hi = mid - 1
    return found
