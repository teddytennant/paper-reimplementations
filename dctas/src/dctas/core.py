"""Formulas and the DCTAS procedure from arXiv 2609.30360.

Logarithms are natural. Bernoulli KL uses the convention 0*log(0) = 0.
The winner is not given a separate cost weight. Pair masses W are unordered
and sum to 1. If both endpoints of a pair are suboptimal and each lists the
other in Gamma, both masses are added on that pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Used only when a suboptimal plug-in arm has alpha_m = 0.
ALPHA_EPS = 1e-12


def bernoulli_kl(x: float, y: float) -> float:
    """Bernoulli KL d(x, y) = x*log(x/y) + (1-x)*log((1-x)/(1-y)).

    Limits at 0 and 1 use 0*log(0) = 0. d(x, x) = 0.
    """
    x = float(x)
    y = float(y)
    if x == y:
        return 0.0
    return _xlog_ratio(x, y) + _xlog_ratio(1.0 - x, 1.0 - y)


def _xlog_ratio(a: float, b: float) -> float:
    """a * log(a/b), with 0*log(0) = 0."""
    if a == 0.0:
        return 0.0
    if a < 0.0 or b <= 0.0:
        return math.inf
    return a * math.log(a / b)


def d(x: float, y: float) -> float:
    """Alias for bernoulli_kl."""
    return bernoulli_kl(x, y)


def _h(u: float) -> float:
    return u - math.log(u)


def _h_inv(y: float) -> float:
    """Inverse of h(u) = u - log(u) on [1, inf). h(1) = 1."""
    if y <= 1.0:
        return 1.0
    u = y + math.log(y)
    for _ in range(80):
        u_new = y + math.log(u)
        if abs(u_new - u) <= 1e-15 * max(1.0, u_new):
            u = u_new
            break
        u = u_new
    for _ in range(8):
        fu = u - math.log(u) - y
        df = 1.0 - 1.0 / u
        u = u - fu / df
    return u


def _tilde_h(x: float, z: float) -> float:
    log_z = math.log(z)
    threshold = _h(1.0 / log_z)
    if x > threshold:
        hinv = _h_inv(x)
        return math.exp(1.0 / hinv) * hinv
    return z * (x - math.log(log_z))


def C(x: float) -> float:
    """Threshold function C(x) from the paper definition."""
    arg = (_h_inv(1.0 + float(x)) + math.log(math.pi**2 / 3.0)) / 2.0
    return 2.0 * _tilde_h(arg, 1.5)


def beta(t: float, delta: float, L: int) -> float:
    """Stopping threshold beta(t, delta) for L arms."""
    ell = L - 1
    return (
        3.0 * ell * math.log(1.0 + math.log(t / ell))
        + ell * C(math.log(ell / delta) / ell)
    )


def pairs(L: int) -> list[tuple[int, int]]:
    """Unordered pairs (i, j) with i < j, in lexicographic order."""
    return [(i, j) for i in range(L) for j in range(i + 1, L)]


def empirical_preference(wins: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Empirical preference matrix. Diagonal is 0.5 and is unused."""
    wins = np.asarray(wins)
    counts = np.asarray(counts)
    L = wins.shape[0]
    p = np.full((L, L), 0.5, dtype=float)
    for i, j in pairs(L):
        n = int(counts[i, j])
        if n <= 0:
            continue
        p[i, j] = float(wins[i, j]) / n
        p[j, i] = 1.0 - p[i, j]
    return p


def condorcet_winner(preference: np.ndarray) -> int | None:
    """Unique arm that strictly beats every other, or None."""
    p = np.asarray(preference, dtype=float)
    L = p.shape[0]
    found = None
    for i in range(L):
        if all(p[i, j] > 0.5 for j in range(L) if j != i):
            if found is not None:
                return None
            found = i
    return found


def plug_in_winner(preference: np.ndarray) -> int:
    """Condorcet winner if one exists, else argmax_i min_{j != i} p[i, j].

    Ties in the min rule are broken by the smallest arm index.
    """
    winner = condorcet_winner(preference)
    if winner is not None:
        return winner
    p = np.asarray(preference, dtype=float)
    L = p.shape[0]
    best_i = 0
    best_min = -math.inf
    for i in range(L):
        gap = min(float(p[i, j]) for j in range(L) if j != i)
        if gap > best_min:
            best_min = gap
            best_i = i
    return best_i


def _elimination_score(p_mn: float, cost_m: float, cost_n: float) -> float:
    """d(p, max(0.5, p)) / (c_m + c_n). Zero when p >= 0.5."""
    if p_mn >= 0.5:
        return 0.0
    return bernoulli_kl(p_mn, 0.5) / (cost_m + cost_n)


@dataclass
class CostFractions:
    """Optimal cost fractions and the induced pull fractions.

    alpha_m is 0 on the winner. gamma[m] is empty on the winner.
    W and pull_alpha are symmetric. The upper triangle of each sums to 1.
    """

    winner: int
    alpha_m: np.ndarray
    gamma: list[tuple[int, ...]]
    W: np.ndarray
    pull_alpha: np.ndarray


def optimal_cost_fractions(preference: np.ndarray, costs: np.ndarray) -> CostFractions:
    """Cost fractions W and pull fractions from the optimal-weight proposition.

    For m other than the winner,
        alpha_m = max_{n != m} d(p[m, n], max(0.5, p[m, n])) / (c[m] + c[n])
        Gamma_m = argmax of that quantity
        w_m = (1/alpha_m) / sum_{n != winner} (1/alpha_n)
    w_m is split equally on pairs (m, j) for j in Gamma_m. The winner receives
    no separate weight. If alpha_m is 0, it is replaced by ALPHA_EPS.
    """
    p = np.asarray(preference, dtype=float)
    c = np.asarray(costs, dtype=float)
    L = p.shape[0]
    if c.shape[0] != L:
        raise ValueError("costs must have one entry per arm")
    winner = plug_in_winner(p)

    alpha_m = np.zeros(L, dtype=float)
    gamma: list[tuple[int, ...]] = [() for _ in range(L)]
    for m in range(L):
        if m == winner:
            continue
        scores = []
        for n in range(L):
            if n == m:
                continue
            scores.append((n, _elimination_score(float(p[m, n]), float(c[m]), float(c[n]))))
        best = max(score for _, score in scores)
        if best == 0.0:
            best = ALPHA_EPS
        alpha_m[m] = best
        gamma[m] = tuple(n for n, score in scores if score == max(s for _, s in scores))

    inv_sum = 0.0
    for n in range(L):
        if n == winner:
            continue
        inv_sum += 1.0 / alpha_m[n]

    W = np.zeros((L, L), dtype=float)
    for m in range(L):
        if m == winner:
            continue
        members = gamma[m]
        share = (1.0 / alpha_m[m]) / inv_sum / len(members)
        for j in members:
            W[m, j] += share
            W[j, m] += share

    raw = np.zeros((L, L), dtype=float)
    denom = 0.0
    for i, j in pairs(L):
        value = W[i, j] / (float(c[i]) + float(c[j]))
        raw[i, j] = value
        raw[j, i] = value
        denom += value
    pull_alpha = np.zeros((L, L), dtype=float)
    if denom > 0.0:
        pull_alpha = raw / denom
    return CostFractions(
        winner=winner,
        alpha_m=alpha_m,
        gamma=gamma,
        W=W,
        pull_alpha=pull_alpha,
    )


def weight_pull_allocation(costs: np.ndarray, preference: np.ndarray) -> np.ndarray:
    """Pull fractions alpha from WeightPullAllocation(costs, empirical P).

    Returns a symmetric matrix. The upper triangle sums to 1.
    """
    return optimal_cost_fractions(preference, costs).pull_alpha


def _arm_loss(arm: int, preference: np.ndarray, counts: np.ndarray) -> float:
    """sum_k N_{arm,k} * d(p_hat[arm,k], max(p_hat[arm,k], 0.5))."""
    total = 0.0
    L = preference.shape[0]
    for k in range(L):
        if k == arm:
            continue
        n = int(counts[arm, k])
        if n <= 0:
            continue
        ph = float(preference[arm, k])
        total += n * bernoulli_kl(ph, max(ph, 0.5))
    return total


def glr(wins: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """GLR matrix Z[i, j].

    Z[i, j] = sum_k N[j, k] d(p_hat[j, k], max(p_hat[j, k], 0.5))
            - sum_k N[i, k] d(p_hat[i, k], max(p_hat[i, k], 0.5))
    N[i, i] = 0 and the diagonal of p_hat is unused. Z[i, j] = -Z[j, i].
    """
    preference = empirical_preference(wins, counts)
    L = preference.shape[0]
    losses = np.array([_arm_loss(i, preference, counts) for i in range(L)], dtype=float)
    return losses[None, :] - losses[:, None]


def chernoff_stop(wins: np.ndarray, counts: np.ndarray, t: float, delta: float) -> int | None:
    """Return i if Z[i, j] > beta(t, delta) for every j != i, else None."""
    Z = glr(wins, counts)
    L = Z.shape[0]
    threshold = beta(t, delta, L)
    for i in range(L):
        if all(Z[i, j] > threshold for j in range(L) if j != i):
            return i
    return None


@dataclass
class DCTASResult:
    winner: int
    duels: int
    total_cost: float
    stopped: bool


def dctas(
    preference: np.ndarray,
    costs: np.ndarray,
    delta: float,
    seed: int = 0,
    max_pulls: int = 20000,
) -> DCTASResult:
    """Run DCTAS and return the selected arm, duel count, and total cost.

    Initialization pulls every unordered pair once. After that, each step
    stores one pull-fraction vector. Before the pull, if any pair has
    N_k < sqrt(t_total), the least-sampled pair is pulled (smallest pair
    index breaks ties). Otherwise the pair minimizing
    N_k - sum of stored allocations, including the current vector, is pulled.
    t_total counts every pull, including initialization. The stopping check
    uses counts after the pull and beta(t_total, delta).
    """
    p = np.asarray(preference, dtype=float)
    c = np.asarray(costs, dtype=float)
    L = int(p.shape[0])
    if c.shape[0] != L:
        raise ValueError("costs must have one entry per arm")
    rng = np.random.default_rng(seed)
    pair_list = pairs(L)
    wins = np.zeros((L, L), dtype=np.int64)
    counts = np.zeros((L, L), dtype=np.int64)
    total_cost = 0.0

    def pull(i: int, j: int) -> None:
        nonlocal total_cost
        if rng.random() < p[i, j]:
            wins[i, j] += 1
        else:
            wins[j, i] += 1
        counts[i, j] += 1
        counts[j, i] += 1
        total_cost += float(c[i] + c[j])

    for i, j in pair_list:
        pull(i, j)
    t_total = len(pair_list)
    cumulative = np.zeros(len(pair_list), dtype=float)

    stopped = False
    winner = None
    while t_total < max_pulls:
        phat = empirical_preference(wins, counts)
        allocation = weight_pull_allocation(c, phat)
        alpha = np.array([allocation[i, j] for i, j in pair_list], dtype=float)
        sampled = np.array([int(counts[i, j]) for i, j in pair_list], dtype=float)
        if np.any(sampled < math.sqrt(t_total)):
            k = int(np.argmin(sampled))
        else:
            k = int(np.argmin(sampled - (cumulative + alpha)))
        cumulative += alpha
        i, j = pair_list[k]
        pull(i, j)
        t_total += 1
        chosen = chernoff_stop(wins, counts, t_total, delta)
        if chosen is not None:
            winner = chosen
            stopped = True
            break

    if winner is None:
        winner = plug_in_winner(empirical_preference(wins, counts))
    return DCTASResult(
        winner=int(winner),
        duels=int(t_total),
        total_cost=float(total_cost),
        stopped=stopped,
    )
