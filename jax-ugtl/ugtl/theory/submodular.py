"""Set-function machinery for Theorems 4 and 5.

Contents:
  * small instance families (Definitions 7 and 8);
  * the submodularity ratio of Definition 6, computed by brute force;
  * the greedy algorithm (Algorithm 1) and local search (Algorithm 2);
  * a surrogate oracle that perturbs the true value by a factor in
    [1 - beta, 1 + beta] at every set the algorithm evaluates, which is exactly
    the condition Definition 4 asks for along the trajectory.

Everything runs on plain Python sets and numpy. The ground sets are tiny so the
optimum can be found by enumeration, which is what makes the bound checks
meaningful.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence

import numpy as np

Subset = frozenset

__all__ = [
    "MaxCoverage",
    "FacilityLocation",
    "PowerModular",
    "MaxCut",
    "submodularity_ratio",
    "greedy",
    "local_search",
    "brute_force_max",
    "PerturbedOracle",
]


class MaxCoverage:
    """f(X) = total weight of the universe elements covered by the sets in X.

    Monotone and submodular, the instance family behind Theorem 1's hardness
    result and the running example of Definition 7.
    """

    def __init__(self, cover: Sequence[Iterable[int]], weights: np.ndarray | None = None):
        self.cover = [frozenset(c) for c in cover]
        self.n = len(self.cover)
        universe = sorted(set().union(*self.cover)) if self.cover else []
        self.universe = universe
        if weights is None:
            weights = np.ones(len(universe))
        self.weights = {u: float(w) for u, w in zip(universe, weights)}

    def __call__(self, X: Iterable[int]) -> float:
        covered: set[int] = set()
        for i in X:
            covered |= self.cover[i]
        return float(sum(self.weights[u] for u in covered))


class FacilityLocation:
    """f(X) = sum_j max_{i in X} W[i, j], with f(empty) = 0.

    Monotone and submodular for non-negative W.
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        if self.W.min() < 0:
            raise ValueError("facility location needs non-negative weights")
        self.n = self.W.shape[0]

    def __call__(self, X: Iterable[int]) -> float:
        idx = list(X)
        if not idx:
            return 0.0
        return float(self.W[idx].max(axis=0).sum())


class PowerModular:
    """f(X) = (sum_{i in X} w_i)^p.

    Monotone and non-negative for w >= 0. For p > 1 the outer power is convex,
    so the function is supermodular and its submodularity ratio of Definition 6
    is strictly below 1. This is the approximately submodular case that makes
    the gamma in Theorem 4 do some work.
    """

    def __init__(self, w: np.ndarray, p: float = 1.3):
        self.w = np.asarray(w, dtype=float)
        if self.w.min() < 0:
            raise ValueError("weights must be non-negative")
        self.p = float(p)
        self.n = len(self.w)

    def __call__(self, X: Iterable[int]) -> float:
        idx = list(X)
        total = float(self.w[idx].sum()) if idx else 0.0
        return total**self.p


class MaxCut:
    """f(X) = sum of edge weights crossing the cut (X, V \\ X).

    Non-negative, submodular and non-monotone, the example Definition 8 names.
    """

    def __init__(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)
        if not np.allclose(self.W, self.W.T):
            raise ValueError("max cut needs a symmetric weight matrix")
        self.n = self.W.shape[0]

    def __call__(self, X: Iterable[int]) -> float:
        inside = np.zeros(self.n, dtype=bool)
        idx = list(X)
        inside[idx] = True
        return float(self.W[np.ix_(inside, ~inside)].sum())


def submodularity_ratio(f: Callable[[Iterable[int]], float], X: Iterable[int], l: int, n: int) -> float:
    """gamma_{X,l}(f) of Definition 6, by enumeration.

        gamma = min over L subset of X, S with |S| <= l and S disjoint from L, of
                sum_{v in S} (f(L + v) - f(L)) / (f(L + S) - f(L))

    Pairs with a non-positive denominator carry no constraint and are skipped.
    The result is capped at 1 because the ratio is defined to live in [0, 1].
    """
    X = list(X)
    V = range(n)
    best = np.inf
    for r in range(len(X) + 1):
        for L in itertools.combinations(X, r):
            L = frozenset(L)
            fL = f(L)
            rest = [v for v in V if v not in L]
            for size in range(1, l + 1):
                for S in itertools.combinations(rest, size):
                    denom = f(L | frozenset(S)) - fL
                    if denom <= 1e-12:
                        continue
                    numer = sum(f(L | {v}) - fL for v in S)
                    best = min(best, numer / denom)
    if not np.isfinite(best):
        return 1.0
    return float(min(best, 1.0))


def brute_force_max(f: Callable[[Iterable[int]], float], n: int, k: int | None = None) -> tuple[frozenset, float]:
    """Optimum of Definition 7 (k given) or Definition 8 (k = None) by enumeration."""
    sizes = range(0, (k if k is not None else n) + 1)
    best_set, best_val = frozenset(), f(frozenset())
    for size in sizes:
        for S in itertools.combinations(range(n), size):
            v = f(frozenset(S))
            if v > best_val:
                best_set, best_val = frozenset(S), v
    return best_set, float(best_val)


class PerturbedOracle:
    """A surrogate f~ with (1 - beta) f(X) <= f~(X) <= (1 + beta) f(X).

    The multiplier for a set is decided the first time the set is queried and
    then cached, so f~ stays a function and the trajectory condition of
    Definition 4 (Eq. 5) holds by construction at every evaluated set.

    Two policies:
      * ``adversarial`` picks the multiplier that hurts the caller the most.
        With an incumbent it inflates the incumbent and deflates the moves, which
        is what stalls local search; without one it deflates the truly best set
        in the batch and inflates the rest, which is what derails greedy.
      * ``random`` flips a fair coin, which is the sweep the seeded tests use.

    ``visited`` records every set the algorithm evaluated, which is the set the
    Definition 4 guarantee ranges over.
    """

    def __init__(self, f: Callable[[Iterable[int]], float], beta: float, policy: str = "adversarial", seed: int = 0):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must lie in [0, 1), got {beta}")
        if policy not in ("adversarial", "random"):
            raise ValueError(f"unknown policy {policy!r}")
        self.f = f
        self.beta = float(beta)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.cache: dict[frozenset, float] = {}
        self.visited: list[frozenset] = []
        self.calls = 0

    def _record(self, X: frozenset, mult: float) -> float:
        if X not in self.cache:
            self.cache[X] = self.f(X) * mult
            self.visited.append(X)
        return self.cache[X]

    def query(self, sets: Sequence[Iterable[int]], incumbent: Iterable[int] | None = None) -> list[float]:
        """Evaluate f~ on a batch of sets, the way one step of an algorithm does."""
        sets = [frozenset(S) for S in sets]
        self.calls += len(sets)
        lo, hi = 1.0 - self.beta, 1.0 + self.beta
        if self.policy == "random":
            mults = [lo if b else hi for b in self.rng.random(len(sets)) < 0.5]
        elif incumbent is not None:
            inc = frozenset(incumbent)
            mults = [hi if S == inc else lo for S in sets]
        else:
            true_vals = [self.f(S) for S in sets]
            worst = int(np.argmax(true_vals))
            mults = [lo if i == worst else hi for i in range(len(sets))]
        return [self._record(S, m) for S, m in zip(sets, mults)]

    def value(self, X: Iterable[int]) -> float:
        return self.query([X])[0]

    def condition_holds(self, tol: float = 1e-9) -> bool:
        """Check Eq. (3) on the whole visited trajectory."""
        for X in self.visited:
            true = self.f(X)
            got = self.cache[X]
            if got < (1.0 - self.beta) * true - tol or got > (1.0 + self.beta) * true + tol:
                return False
        return True


def greedy(oracle, n: int, k: int) -> frozenset:
    """Algorithm 1, run on the surrogate f~ that ``oracle`` exposes.

    Lines 3 and 4: take the element with the largest surrogate value of
    X_i + {v}, k times. Every set it scores counts as visited for Definition 4.
    """
    X = frozenset()
    for _ in range(k):
        candidates = [v for v in range(n) if v not in X]
        if not candidates:
            break
        sets = [X | {v} for v in candidates]
        vals = oracle.query(sets)
        X = sets[int(np.argmax(vals))]
    return X


def local_search(oracle, n: int, eps: float, max_iters: int = 10_000) -> frozenset:
    """Algorithm 2, run on the surrogate f~ that ``oracle`` exposes.

    Line 1 starts from the best single element. Lines 2 to 4 add and lines 5 to 7
    delete while some move improves the surrogate value by a factor larger than
    1 + eps / n^2. Line 8 returns the better of X and V \\ X, judged by f~, which
    is the step the last display of the Theorem 5 proof uses.
    """
    singletons = [frozenset({v}) for v in range(n)]
    vals = oracle.query(singletons)
    X = singletons[int(np.argmax(vals))]
    threshold = 1.0 + eps / n**2

    for _ in range(max_iters):
        fx = oracle.query([X], incumbent=X)[0]
        adds = [X | {v} for v in range(n) if v not in X]
        if adds:
            vals = oracle.query(adds, incumbent=X)
            best = int(np.argmax(vals))
            if vals[best] > threshold * fx:
                X = adds[best]
                continue
        dels = [X - {v} for v in X]
        if dels:
            vals = oracle.query(dels, incumbent=X)
            best = int(np.argmax(vals))
            if vals[best] > threshold * fx:
                X = dels[best]
                continue
        break
    else:  # pragma: no cover - the factor-(1 + eps/n^2) increase bounds the count
        raise RuntimeError("local search did not converge")

    complement = frozenset(range(n)) - X
    both = [X, complement]
    vals = oracle.query(both)
    return both[int(np.argmax(vals))]
