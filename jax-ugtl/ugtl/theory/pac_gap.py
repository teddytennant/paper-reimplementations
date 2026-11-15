"""The instance behind Section 3.1: accurate almost everywhere, useless anyway.

Theorem 1 says maximum coverage is PMAC learnable and optimizable and still not
offline optimizable. The mechanism the paper describes is that the parts of the
instance that determine the optimum are the parts the samples cannot resolve,
while everything else is learned fine. This module builds a small explicit
instance with that shape, so the failure can be run rather than cited.

The ground set holds n - k decoys, each covering one universe element, and k
jackpots, each covering ``block`` elements. The optimum under a budget of k is
to take the jackpots, worth k * block. The surrogate counts a jackpot as
covering one element, so it is exactly right on every subset with no jackpot in
it, which is nearly all of them under a uniform draw when k^2 << n, and it is
wrong by a factor of ``block`` precisely on the optimal region.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["HiddenJackpotCoverage"]


class HiddenJackpotCoverage:
    def __init__(self, n: int = 500, k: int = 3, block: int = 100):
        if k >= n:
            raise ValueError("need more elements than the budget")
        self.n, self.k, self.block = n, k, block
        self.n_decoys = n - k
        self.jackpots = frozenset(range(self.n_decoys, n))

    def f(self, X: Iterable[int]) -> float:
        """The true coverage value."""
        X = frozenset(X)
        return float(len(X - self.jackpots) + self.block * len(X & self.jackpots))

    def surrogate(self, X: Iterable[int]) -> float:
        """The learned function: every jackpot looks like one more decoy."""
        return float(len(frozenset(X)))

    @property
    def optimum(self) -> float:
        """max_{|X| <= k} f(X), which is to take every jackpot."""
        return float(self.k * self.block)

    def mismatch_rate(self, rng, samples: int = 20_000) -> float:
        """Fraction of uniform size-k subsets where the surrogate is wrong.

        This is the epsilon in the PMAC statement of Definition 2 for this
        surrogate under the uniform distribution over feasible subsets.
        """
        idx = np.array([rng.choice(self.n, size=self.k, replace=False) for _ in range(samples)])
        hits = (idx >= self.n_decoys).any(axis=1)
        return float(hits.mean())
