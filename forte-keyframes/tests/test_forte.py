"""Behavior checks for coverage, the banded GP, acquisition, and exact DP."""

from itertools import combinations

import numpy as np
import pytest

from forte import (
    adaptive_score,
    coverage,
    exact_keyframes,
    fit_gp,
    marginal_gain,
    predict,
    select_keyframes,
)


def _naive_dp(times, scores, k, horizon, beta):
    """Independent O(N^2) recurrence, Equations (17)-(19)."""
    n = len(times)
    tau = horizon / (k + 1.0)
    values = np.full((k + 1, n), -np.inf)
    prev = np.full((k + 1, n), -1, dtype=int)
    for i in range(n):
        values[1, i] = scores[i] + beta * np.log1p(times[i] / tau)
    for count in range(2, k + 1):
        for i in range(count - 1, n):
            best = -np.inf
            best_j = -1
            for j in range(count - 2, i):
                step = values[count - 1, j] + beta * np.log1p(
                    (times[i] - times[j]) / tau
                )
                if step > best:
                    best = step
                    best_j = j
            values[count, i] = scores[i] + best
            prev[count, i] = best_j
    best_f = -np.inf
    best_i = -1
    for i in range(k - 1, n):
        terminal = values[k, i] + beta * np.log1p((horizon - times[i]) / tau)
        if terminal > best_f:
            best_f = terminal
            best_i = i
    chosen = [best_i]
    cursor = best_i
    for count in range(k, 1, -1):
        cursor = int(prev[count, cursor])
        chosen.append(cursor)
    chosen.reverse()
    return tuple(chosen), float(best_f)


def _max_gap(indices, times, horizon):
    selected = np.concatenate([[0.0], np.sort(times[list(indices)]), [horizon]])
    return float(np.max(np.diff(selected)))


def test_uniform_spacing_maximizes_coverage_and_gains_diminish():
    horizon = 12.0
    tau = 1.5
    count = 5
    uniform = [i * horizon / (count + 1) for i in range(1, count + 1)]
    clustered = [0.2, 0.35, 0.5, 0.7, 0.9]
    assert coverage([], horizon, tau) == pytest.approx(np.log1p(horizon / tau))
    assert coverage([0.5], 1.0, 1.0) == pytest.approx(2.0 * np.log1p(0.5))
    assert coverage(uniform, horizon, tau) > coverage(clustered, horizon, tau)

    perturbed = list(uniform)
    perturbed[1] += 0.4
    assert coverage(uniform, horizon, tau) > coverage(perturbed, horizon, tau)

    expected = np.log1p(4.0 / 2.0) + np.log1p(6.0 / 2.0) - np.log1p(10.0 / 2.0)
    gain = marginal_gain(4.0, [], 10.0, 2.0)
    assert gain == pytest.approx(expected)
    assert gain >= 0.0

    selected = [1.0, 5.0, 9.0]
    for point in (0.4, 3.0, 7.5, 11.0):
        direct = coverage(selected + [point], horizon, tau) - coverage(
            selected, horizon, tau
        )
        assert marginal_gain(point, selected, horizon, tau) == pytest.approx(direct)
        assert direct >= -1e-12

    smaller = [3.0]
    larger = [3.0, 7.0]
    point = 5.0
    assert marginal_gain(point, smaller, 10.0, 1.0) >= marginal_gain(
        point, larger, 10.0, 1.0
    ) - 1e-12
    assert marginal_gain(3.0, larger, 10.0, 1.0) == pytest.approx(0.0)


def test_exact_dp_matches_brute_force_and_beta_zero():
    times = np.array([0.5, 1.4, 2.6, 3.7, 4.9, 6.2, 7.3, 8.5, 9.4])
    scores = np.array([0.15, 0.92, 0.08, 0.71, 0.33, 0.88, 0.12, 0.64, 0.27])
    horizon = 10.0
    beta = 1.1
    k = 3
    tau = horizon / (k + 1.0)

    best_set = None
    best_f = -np.inf
    near = 0
    for comb in combinations(range(len(times)), k):
        chosen = list(comb)
        value = float(scores[chosen].sum() + beta * coverage(times[chosen], horizon, tau))
        if value > best_f + 1e-8:
            best_f = value
            best_set = comb
            near = 1
        elif abs(value - best_f) <= 1e-8:
            near += 1
    assert near == 1

    indices, objective = exact_keyframes(times, scores, k, horizon, beta)
    assert indices == best_set
    assert objective == pytest.approx(best_f, abs=1e-8)
    recomputed = float(
        scores[list(indices)].sum() + beta * coverage(times[list(indices)], horizon, tau)
    )
    assert objective == pytest.approx(recomputed, abs=1e-8)

    k1 = 1
    tau1 = horizon / (k1 + 1.0)
    best_i = None
    best_f1 = -np.inf
    for i in range(len(times)):
        value = float(scores[i] + beta * coverage([times[i]], horizon, tau1))
        if value > best_f1:
            best_f1 = value
            best_i = (i,)
    indices1, objective1 = exact_keyframes(times, scores, k1, horizon, beta)
    assert indices1 == best_i
    assert objective1 == pytest.approx(best_f1, abs=1e-8)

    zero_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    zero_scores = np.array([0.2, 0.9, 0.4, 0.8, 0.1])
    zero_idx, zero_obj = exact_keyframes(zero_times, zero_scores, 2, 6.0, 0.0)
    assert zero_idx == (1, 3)
    assert zero_obj == pytest.approx(1.7)


def test_deque_matches_naive_dp_on_index_set():
    times = np.linspace(0.4, 18.6, 20)
    scores = np.array(
        [
            0.12, 0.85, 0.33, 0.91, 0.07, 0.62, 0.44, 0.78, 0.19, 0.55,
            0.96, 0.28, 0.71, 0.14, 0.83, 0.39, 0.67, 0.22, 0.58, 0.47,
        ]
    )
    horizon = 20.0
    k = 4
    beta = 0.9
    naive_idx, naive_obj = _naive_dp(times, scores, k, horizon, beta)
    exact_idx, exact_obj = exact_keyframes(times, scores, k, horizon, beta)
    assert exact_idx == naive_idx
    assert exact_obj == pytest.approx(naive_obj, abs=1e-8)
    tau = horizon / (k + 1.0)
    formula = float(
        scores[list(exact_idx)].sum()
        + beta * coverage(times[list(exact_idx)], horizon, tau)
    )
    assert exact_obj == pytest.approx(formula, abs=1e-8)


def test_banded_gp_matches_dense_and_recovers():
    times = np.linspace(0.0, 12.0, 16)
    scores = np.sin(times / 2.0)
    length_scale = 2.0
    queries = np.linspace(0.0, 12.0, 40)

    wide = fit_gp(times, scores, length_scale, bandwidth=len(times) - 1)
    assert wide.bandwidth == len(times) - 1
    predicted = np.asarray(predict(wide, queries))

    delta = times[:, None] - times[None, :]
    kernel = np.exp(-0.5 * (delta / length_scale) ** 2)
    mu_obs = float(np.mean(scores))
    alpha = np.linalg.solve(
        kernel + wide.eps * np.eye(len(times)), scores - mu_obs
    )
    reference = []
    for query in queries:
        row = np.exp(-0.5 * ((query - times) / length_scale) ** 2)
        reference.append(mu_obs + row @ alpha)
    assert np.allclose(predicted, np.asarray(reference), rtol=1e-6, atol=1e-6)

    dense = fit_gp(times, scores, length_scale, dense=True)
    dense_pred = np.asarray(predict(dense, queries))
    assert np.allclose(dense_pred, np.asarray(reference), rtol=1e-6, atol=1e-6)

    far = float(np.asarray(predict(wide, np.array([1.0e6])))[0])
    assert far == pytest.approx(mu_obs, abs=1e-6)

    smooth_times = np.linspace(0.0, 20.0, 21)
    smooth_scores = np.exp(-0.5 * ((smooth_times - 10.0) / 2.0) ** 2)
    narrow = fit_gp(smooth_times, smooth_scores, length_scale=0.3, bandwidth=1)
    narrow_pred = np.asarray(predict(narrow, smooth_times))
    assert np.all(np.isfinite(narrow_pred))

    hard_times = np.linspace(0.0, 9.0, 10)
    hard_scores = np.linspace(0.0, 1.0, 10)
    recovered = fit_gp(hard_times, hard_scores, length_scale=10.0, bandwidth=1)
    assert any(not success for _, _, success in recovered.attempts)
    assert recovered.widened or recovered.eps > 1e-12
    recovered_pred = np.asarray(predict(recovered, np.linspace(0.0, 9.0, 15)))
    assert np.all(np.isfinite(recovered_pred))


def test_acquisition_concentrates_or_spreads_and_respects_budget():
    horizon = 100.0
    times = np.linspace(0.0, horizon, 101)
    budget = 9
    n_init = 5
    region = (40.0, 60.0)

    def peaked(t):
        return 1.0 if region[0] <= t <= region[1] else 0.0

    def flat(t):
        return 1.0

    peaked_set = adaptive_score(
        times,
        peaked,
        budget,
        horizon,
        beta_acq=0.0,
        length_scale=3.0,
        n_init=n_init,
        n_rounds=4,
    )
    flat_set = adaptive_score(
        times,
        flat,
        budget,
        horizon,
        beta_acq=2.0,
        length_scale=3.0,
        n_init=n_init,
        n_rounds=4,
    )
    assert len(peaked_set.indices) == budget
    assert len(flat_set.indices) == budget

    init = set(np.rint(np.linspace(0, len(times) - 1, n_init)).astype(int))
    extras = [i for i in peaked_set.indices if i not in init]
    assert extras
    assert all(region[0] <= float(times[i]) <= region[1] for i in extras)
    assert _max_gap(flat_set.indices, times, horizon) < _max_gap(
        peaked_set.indices, times, horizon
    )


def test_end_to_end_objective_matches_equation_five():
    horizon = 20.0
    times = np.linspace(0.0, horizon, 41)
    budget = 12
    k = 4
    beta_sel = 0.8

    def scorer(t):
        return float(np.sin(t / 3.0) + 0.05 * t)

    result = select_keyframes(
        times,
        scorer,
        budget,
        k,
        horizon,
        beta_acq=1.0,
        beta_sel=beta_sel,
        length_scale=3.0,
        n_init=6,
        n_rounds=3,
    )
    assert len(result.indices) == k
    assert set(result.indices).issubset(set(result.pool_indices))
    assert result.indices == tuple(sorted(result.indices))

    raw = np.array([scorer(float(times[i])) for i in result.pool_indices])
    lo = float(raw.min())
    hi = float(raw.max())
    assert hi > lo
    normalized = (raw - lo) / (hi - lo)
    positions = [result.pool_indices.index(i) for i in result.indices]
    selected_scores = normalized[positions]
    tau = horizon / (k + 1.0)
    objective = float(
        selected_scores.sum()
        + beta_sel * coverage(times[list(result.indices)], horizon, tau)
    )
    assert result.objective == pytest.approx(objective, abs=1e-8)
