"""Algorithms 1–3: coordinate-wise monotonicity and DSIC testers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ic_tester.dodis import sample_dodis_pair
from ic_tester.lines import assemble_bids, binary_search_first, eval_line

# Repeat constant for Algorithm 1. The paper writes O(n log m / ε) and
# internally ceil(2 C n log(m) / ε) for a Dodis-lemma constant C. This
# kernel uses n_repeats = ceil(C * n * ln(m) / ε) with C = 8.
ALG1_C = 8

AllocFn = Callable[[NDArray[np.int_]], ArrayLike]
Mechanism1D = Callable[[int], tuple[float, float]]
MechanismND = Callable[[NDArray[np.int_]], tuple[ArrayLike, ArrayLike]]
SampleBids = Callable[[np.random.Generator], ArrayLike]
SampleLineBid = Callable[[np.random.Generator, ArrayLike, int, ArrayLike], int]


@dataclass(frozen=True)
class TesterResult:
    """Outcome of a property tester.

    ``n_repeats`` is the configured trial budget (not the number executed
    before a possible early reject). ``n_queries`` counts oracle calls.
    """

    accepted: bool
    n_queries: int
    n_repeats: int


def default_alg1_repeats(
    n: int,
    m: int,
    eps: float,
    C: float = ALG1_C,
) -> int:
    """Trial budget for Algorithm 1: ceil(C * n * ln(m) / ε)."""
    return math.ceil(C * n * math.log(m) / eps)


class _Cached1D:
    """Query-caching wrapper for a 1D mechanism a(b) = (x(b), p(b))."""

    def __init__(self, a: Mechanism1D):
        self._a = a
        self.n_queries = 0
        self._cache: dict[int, tuple[float, float]] = {}

    def query(self, b: int) -> tuple[float, float]:
        b = int(b)
        if b not in self._cache:
            x_b, p_b = self._a(b)
            self._cache[b] = (float(x_b), float(p_b))
            self.n_queries += 1
        return self._cache[b]

    def x(self, b: int) -> int:
        return int(self.query(b)[0])

    def p(self, b: int) -> float:
        return float(self.query(b)[1])


def coordinatewise_monotonicity(
    x: AllocFn,
    n: int,
    m: int,
    eps: float,
    rng: np.random.Generator,
    n_repeats: Optional[int] = None,
    C: float = ALG1_C,
) -> TesterResult:
    """Algorithm 1: one-sided non-adaptive coordinate-wise monotonicity tester.

    Each trial draws i uniform in {0,...,n-1}, prefixes/suffixes uniform
    on [0:m], and a Dodis pair (p, l) ~ 𝒟, then rejects if the i-line
    decreases from p to l. Two allocation queries per trial.
    """
    if not (0 < eps <= 1):
        raise ValueError(f"eps must be in (0, 1], got {eps}")
    if n_repeats is None:
        n_repeats = default_alg1_repeats(n, m, eps, C=C)

    n_queries = 0

    def counted_x(bids: NDArray[np.int_]) -> ArrayLike:
        nonlocal n_queries
        n_queries += 1
        return x(bids)

    for _ in range(n_repeats):
        i = int(rng.integers(0, n))
        alpha = rng.integers(0, m + 1, size=i)
        beta = rng.integers(0, m + 1, size=n - 1 - i)
        p, l = sample_dodis_pair(rng, m)
        x_p = eval_line(counted_x, alpha, i, beta, p)
        x_l = eval_line(counted_x, alpha, i, beta, l)
        if x_p > x_l:
            return TesterResult(accepted=False, n_queries=n_queries, n_repeats=n_repeats)
    return TesterResult(accepted=True, n_queries=n_queries, n_repeats=n_repeats)


def dsic_1d(
    a: Mechanism1D,
    m: int,
    d: int,
    eps: float,
    rng: np.random.Generator,
    sample_bid: Optional[Callable[[np.random.Generator], int]] = None,
    atol: float = 1e-9,
) -> TesterResult:
    """Algorithm 2: one-sided adaptive 1D DSIC line tester.

    Binary-searches each allocation level ℓ ∈ {0,...,d}, checks that the
    recovered jumps start at 0 and induce Myerson multipliers h_i in
    [t_i-1, t_i], reconstructs the monotone (y, p_y), then samples
    ceil(2/ε) bids from D' (default: uniform on {0,...,m}).
    """
    if not (0 < eps <= 1):
        raise ValueError(f"eps must be in (0, 1], got {eps}")
    n_repeats = math.ceil(2.0 / eps)
    if sample_bid is None:
        sample_bid = lambda r: int(r.integers(0, m + 1))

    oracle = _Cached1D(a)

    # Steps 1–5: recover candidate jumps via binary search.
    S_levels: list[int] = []
    v: dict[int, int] = {}
    for ell in range(d + 1):
        found = binary_search_first(oracle.x, m, ell)
        if found is not None:
            v[ell] = found
            S_levels.append(ell)

    if not S_levels:
        return TesterResult(accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats)

    S_levels.sort()
    t = [v[s] for s in S_levels]
    if t[0] != 0:
        return TesterResult(accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats)

    k = len(t) - 1
    h: list[float] = []
    for i in range(1, k + 1):
        if t[i - 1] >= t[i]:
            return TesterResult(
                accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats
            )
        x_i = oracle.x(t[i])
        x_prev = oracle.x(t[i - 1])
        dx = x_i - x_prev
        if dx == 0:
            return TesterResult(
                accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats
            )
        hi = (oracle.p(t[i]) - oracle.p(t[i - 1])) / float(dx)
        # Reject unless t_i - 1 <= h_i <= t_i.
        if hi > t[i] + atol or hi < (t[i] - 1) - atol:
            return TesterResult(
                accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats
            )
        h.append(float(hi))

    # Steps 6–7: reconstruct monotone y and valid p_y from (t, h).
    # y(b) = x(t_b) with t_b the largest recorded jump <= b.
    y_table = np.empty(m + 1, dtype=np.int64)
    py_table = np.empty(m + 1, dtype=np.float64)
    x_at_t = [oracle.x(ti) for ti in t]
    price = 0.0
    for i in range(k + 1):
        start = t[i]
        end = t[i + 1] if i < k else m + 1
        if i >= 1:
            price += h[i - 1] * (x_at_t[i] - x_at_t[i - 1])
        y_table[start:end] = x_at_t[i]
        py_table[start:end] = price

    # Step 8: disagreement sampling under D'.
    for _ in range(n_repeats):
        b = int(sample_bid(rng))
        x_b, p_b = oracle.query(b)
        if int(x_b) != int(y_table[b]) or abs(float(p_b) - float(py_table[b])) > atol:
            return TesterResult(
                accepted=False, n_queries=oracle.n_queries, n_repeats=n_repeats
            )
    return TesterResult(accepted=True, n_queries=oracle.n_queries, n_repeats=n_repeats)


def dsic(
    a: MechanismND,
    n: int,
    m: int,
    d: int,
    eps: float,
    rng: np.random.Generator,
    sample_bids: Optional[SampleBids] = None,
    sample_line_bid: Optional[SampleLineBid] = None,
    atol: float = 1e-9,
) -> TesterResult:
    """Algorithm 3: one-sided n-dimensional DSIC tester.

    Default D is uniform on [0:m]^n. ``sample_bids(rng)`` and
    ``sample_line_bid(rng, alpha, i, beta)`` may replace D and its
    1D conditionals D_{α,i,β}.
    """
    if not (0 < eps <= 1):
        raise ValueError(f"eps must be in (0, 1], got {eps}")
    if sample_bids is None:
        sample_bids = lambda r: r.integers(0, m + 1, size=n)
    if sample_line_bid is None:
        sample_line_bid = lambda r, alpha, i, beta: int(r.integers(0, m + 1))

    eps_prime = eps / (2.0 * n)
    ell = math.ceil(math.log2(2.0 / eps_prime))
    planned = 0
    for j in range(1, ell + 1):
        planned += math.ceil(12.0 * ell / ((2**j) * eps_prime))

    n_queries = 0
    for j in range(1, ell + 1):
        inner = math.ceil(12.0 * ell / ((2**j) * eps_prime))
        dist = 2.0 ** (-j)
        for _ in range(inner):
            b = np.asarray(sample_bids(rng), dtype=int).reshape(-1)
            i = int(rng.integers(0, n))
            alpha = b[:i]
            beta = b[i + 1 :]

            def line_a(bid: int, _alpha=alpha, _i=i, _beta=beta):
                bids = assemble_bids(_alpha, bid, _beta)
                x_vec, p_vec = a(bids)
                return float(np.asarray(x_vec)[_i]), float(np.asarray(p_vec)[_i])

            def line_sample(r, _alpha=alpha, _i=i, _beta=beta):
                return int(sample_line_bid(r, _alpha, _i, _beta))

            for _run in range(2):
                result = dsic_1d(
                    line_a,
                    m=m,
                    d=d,
                    eps=dist,
                    rng=rng,
                    sample_bid=line_sample,
                    atol=atol,
                )
                n_queries += result.n_queries
                if not result.accepted:
                    return TesterResult(
                        accepted=False, n_queries=n_queries, n_repeats=planned
                    )
    return TesterResult(accepted=True, n_queries=n_queries, n_repeats=planned)
