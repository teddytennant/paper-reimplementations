"""Acquisition scoring and exact keyframe optimization, Algorithm 1.

The scorer is an injected callable psi(t) -> float. No vision model is run.
"""

from collections import deque
from dataclasses import dataclass
from math import expm1, log1p

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from forte.coverage import marginal_gains
from forte.gp import EPS_SCHEDULE, fit_gp, minmax_normalize, predict

DEFAULT_BETA_ACQ = 3.5
DEFAULT_BETA_SEL = 1.0
DEFAULT_LENGTH_SCALE = 10.0
DEFAULT_N_ROUNDS = 4


@dataclass(frozen=True)
class ScoreResult:
    """Frames scored by the adaptive rule, in timestamp order."""

    indices: tuple
    raw_scores: jnp.ndarray
    normalized_scores: jnp.ndarray


@dataclass(frozen=True)
class KeyframeResult:
    """Selected keyframes and the Equation (5) objective.

    indices are positions in the timestamp universe, sorted in time, and lie
    in the scored pool. pool_scores are the normalized relevance values used
    as r_i in Equation (5), aligned with pool_indices.
    """

    indices: tuple
    objective: float
    pool_indices: tuple
    pool_scores: jnp.ndarray
    raw_scores: jnp.ndarray


def _finite_times(timestamps):
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps must be nonempty")
    if not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be finite")
    if times.size >= 2 and np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return times


def _intersection(t_p, t_q, d_p, d_q, tau, beta):
    """Intersection X(p, q) of two coverage lines.

    Denominator is -expm1(-(d_q - d_p) / beta), not a direct large exp.
    """
    diff = (d_q - d_p) / beta
    denom = -expm1(-diff)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.inf
    return t_p - tau + (t_q - t_p) / denom


def _line(d_j, t_j, x, tau, beta):
    """V_j(x) = d_j + beta * log1p((x - t_j) / tau)."""
    return d_j + beta * log1p((x - t_j) / tau)


def _top_k_indices(scores, n_keyframes):
    """K highest scores, earlier index first on ties, then temporal order."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    chosen = np.sort(order[:n_keyframes])
    return tuple(int(i) for i in chosen)


def exact_keyframes(timestamps, scores, n_keyframes, T, beta):
    """Exact maximizer of Equation (5) via Algorithm 1.

    Returns pool indices in temporal order and F*. For beta = 0 the coverage
    term is zero, so the result is the K highest-scoring indices in time order
    and the objective is the sum of those scores.

    Floating-point ties can differ from real-arithmetic optimality. Callers
    that need a unique optimum should use well-separated timestamps and scores.
    """
    times = _finite_times(timestamps)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != times.shape:
        raise ValueError("scores must match timestamps")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    horizon = float(T)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("T must be positive")
    if np.any(times < 0.0) or np.any(times > horizon):
        raise ValueError("timestamps must lie in [0, T]")
    weight = float(beta)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("beta must be nonnegative")
    n = int(times.shape[0])
    k = int(n_keyframes)
    if k != n_keyframes or k < 1 or k > n:
        raise ValueError("n_keyframes must be an integer in 1..N")
    if weight == 0.0:
        chosen = _top_k_indices(values, k)
        objective = float(np.sum(values[list(chosen)]))
        return chosen, objective

    tau = horizon / (k + 1.0)
    prev = values + weight * np.log1p(times / tau)
    predecessor = np.full((k + 1, n), -1, dtype=np.int64)
    for count in range(2, k + 1):
        current = np.full(n, -np.inf, dtype=np.float64)
        envelope = deque()
        running_max = -np.inf
        for i in range(count - 1, n):
            pred = i - 1
            pred_value = float(prev[pred])
            if pred_value > running_max:
                running_max = pred_value
                while len(envelope) >= 2 and _intersection(
                    float(times[envelope[-2]]),
                    float(times[envelope[-1]]),
                    float(prev[envelope[-2]]),
                    float(prev[envelope[-1]]),
                    tau,
                    weight,
                ) >= _intersection(
                    float(times[envelope[-1]]),
                    float(times[pred]),
                    float(prev[envelope[-1]]),
                    pred_value,
                    tau,
                    weight,
                ):
                    envelope.pop()
                envelope.append(pred)
            if not envelope:
                raise RuntimeError("keyframe envelope was empty")
            query = float(times[i])
            while len(envelope) >= 2 and _line(
                float(prev[envelope[1]]), float(times[envelope[1]]), query, tau, weight
            ) >= _line(
                float(prev[envelope[0]]), float(times[envelope[0]]), query, tau, weight
            ):
                envelope.popleft()
            best = int(envelope[0])
            current[i] = values[i] + _line(
                float(prev[best]), float(times[best]), query, tau, weight
            )
            predecessor[count, i] = best
        prev = current

    best_value = -np.inf
    best_end = -1
    for i in range(k - 1, n):
        terminal = float(prev[i]) + weight * log1p((horizon - float(times[i])) / tau)
        if terminal > best_value:
            best_value = terminal
            best_end = i
    if best_end < 0 or not np.isfinite(best_value):
        raise RuntimeError("keyframe DP did not produce a finite optimum")
    chosen_rev = [best_end]
    cursor = best_end
    for count in range(k, 1, -1):
        cursor = int(predecessor[count, cursor])
        if cursor < 0:
            raise RuntimeError("keyframe backtrack failed")
        chosen_rev.append(cursor)
    chosen = tuple(reversed(chosen_rev))
    if len(chosen) != k or chosen != tuple(sorted(chosen)):
        raise RuntimeError("keyframe backtrack returned an invalid index set")
    return chosen, float(best_value)


def uniform_indices(n_frames, n_init):
    """Evenly spaced frame indices, including both endpoints when n_init > 1."""
    n = int(n_frames)
    count = int(n_init)
    if count < 1 or count > n:
        raise ValueError("n_init must lie in 1..n_frames")
    if count == n:
        return np.arange(n, dtype=np.int64)
    raw = np.rint(np.linspace(0, n - 1, count)).astype(np.int64)
    used = set()
    chosen = []
    for seed in raw:
        if int(seed) not in used:
            used.add(int(seed))
            chosen.append(int(seed))
            continue
        for offset in range(1, n):
            placed = False
            for candidate in (int(seed) - offset, int(seed) + offset):
                if 0 <= candidate < n and candidate not in used:
                    used.add(candidate)
                    chosen.append(candidate)
                    placed = True
                    break
            if placed:
                break
    if len(chosen) != count:
        raise RuntimeError("failed to place uniform initialization indices")
    return np.array(sorted(chosen), dtype=np.int64)


def _batch_sizes(remaining, n_rounds):
    rounds = int(n_rounds)
    if rounds < 1:
        raise ValueError("n_rounds must be positive")
    left = int(remaining)
    if left <= 0:
        return []
    base, extra = divmod(left, rounds)
    return [base + (1 if i < extra else 0) for i in range(rounds)]


def adaptive_score(
    timestamps,
    scorer,
    budget,
    T,
    beta_acq=DEFAULT_BETA_ACQ,
    length_scale=DEFAULT_LENGTH_SCALE,
    n_init=None,
    n_rounds=DEFAULT_N_ROUNDS,
    clip_predictions=True,
):
    """Score B frames with Equation (4), updating coverage inside each batch.

    GP predictions stay fixed inside a batch and are refit between batches.
    Initialization uses evenly spaced indices in the frame universe.
    """
    times = _finite_times(timestamps)
    horizon = float(T)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("T must be positive")
    if np.any(times < 0.0) or np.any(times > horizon):
        raise ValueError("timestamps must lie in [0, T]")
    n = int(times.shape[0])
    total = int(budget)
    if total != budget or total < 1 or total > n:
        raise ValueError("budget must be an integer in 1..N")
    weight = float(beta_acq)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("beta_acq must be nonnegative")
    init_count = total // 2 if n_init is None else int(n_init)
    if n_init is not None and init_count != n_init:
        raise ValueError("n_init must be an integer")
    if init_count < 1 or init_count > total:
        raise ValueError("n_init must lie in 1..budget")
    tau = horizon / (total + 1.0)
    raw = np.full(n, np.nan, dtype=np.float64)
    observed = set()
    for index in uniform_indices(n, init_count):
        raw[index] = float(scorer(float(times[index])))
        observed.add(int(index))
    if not np.all(np.isfinite(raw[list(observed)])):
        raise ValueError("scorer must return finite values")

    for batch in _batch_sizes(total - init_count, n_rounds):
        if batch <= 0:
            continue
        obs_idx = np.array(sorted(observed), dtype=np.int64)
        normalized = np.asarray(minmax_normalize(raw[obs_idx]), dtype=np.float64)
        fit = fit_gp(times[obs_idx], normalized, length_scale)
        mu = np.asarray(predict(fit, times), dtype=np.float64)
        if clip_predictions:
            mu = np.clip(mu, float(np.min(normalized)), float(np.max(normalized)))
        active = set(observed)
        chosen = []
        for _ in range(batch):
            candidates = np.array(
                [i for i in range(n) if i not in active], dtype=np.int64
            )
            if candidates.size == 0:
                break
            selected_times = times[np.array(sorted(active), dtype=np.int64)]
            gains = np.asarray(
                marginal_gains(times[candidates], selected_times, horizon, tau),
                dtype=np.float64,
            )
            acquisition = mu[candidates] + weight * gains
            best = int(candidates[int(np.argmax(acquisition))])
            active.add(best)
            chosen.append(best)
        for index in chosen:
            raw[index] = float(scorer(float(times[index])))
            observed.add(index)
        if not np.all(np.isfinite(raw[chosen])):
            raise ValueError("scorer must return finite values")

    if len(observed) != total:
        raise RuntimeError("adaptive scoring did not meet the budget")
    order = tuple(sorted(observed))
    raw_scores = jnp.asarray(raw[list(order)], dtype=jnp.float64)
    return ScoreResult(
        indices=order,
        raw_scores=raw_scores,
        normalized_scores=minmax_normalize(raw_scores),
    )


def select_keyframes(
    timestamps,
    scorer,
    budget,
    n_keyframes,
    T,
    beta_acq=DEFAULT_BETA_ACQ,
    beta_sel=DEFAULT_BETA_SEL,
    length_scale=DEFAULT_LENGTH_SCALE,
    n_init=None,
    n_rounds=DEFAULT_N_ROUNDS,
):
    """Score a pool, then return K keyframes maximizing Equation (5).

    Measured scores are min-max normalized once on the finished pool. Those
    normalized values are the r_i used by exact keyframe optimization, and
    they stay fixed during the DP.
    """
    k = int(n_keyframes)
    if k != n_keyframes or k < 1:
        raise ValueError("n_keyframes must be a positive integer")
    if k > int(budget):
        raise ValueError("n_keyframes cannot exceed the scoring budget")
    scored = adaptive_score(
        timestamps,
        scorer,
        budget,
        T,
        beta_acq=beta_acq,
        length_scale=length_scale,
        n_init=n_init,
        n_rounds=n_rounds,
    )
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    pool = scored.indices
    pool_times = times[list(pool)]
    pool_scores = np.asarray(scored.normalized_scores, dtype=np.float64)
    local, objective = exact_keyframes(
        pool_times, pool_scores, k, T, beta_sel
    )
    universe = tuple(int(pool[i]) for i in local)
    return KeyframeResult(
        indices=universe,
        objective=float(objective),
        pool_indices=pool,
        pool_scores=scored.normalized_scores,
        raw_scores=scored.raw_scores,
    )


# Re-export so callers can see the jitter schedule used between batches.
__all__ = [
    "DEFAULT_BETA_ACQ",
    "DEFAULT_BETA_SEL",
    "DEFAULT_LENGTH_SCALE",
    "DEFAULT_N_ROUNDS",
    "EPS_SCHEDULE",
    "KeyframeResult",
    "ScoreResult",
    "adaptive_score",
    "exact_keyframes",
    "select_keyframes",
    "uniform_indices",
]
