"""Checks for the DCTAS formulas and a small easy instance."""

import math

import numpy as np

from dctas import C, beta, d, dctas, glr, optimal_cost_fractions


def _kl(x: float, y: float) -> float:
    return x * math.log(x / y) + (1.0 - x) * math.log((1.0 - x) / (1.0 - y))


def test_threshold_fixtures():
    assert abs(C(1.0) - 9.1045170257) < 1e-6
    assert abs(C(2.0) - 10.8684870571) < 1e-6
    assert abs(beta(100, 0.1, L=3) - 29.599296378) < 1e-4


def test_kl_matches_formula():
    expected = _kl(0.2, 0.5)
    assert abs(d(0.2, 0.5) - expected) < 1e-12
    assert d(0.8, max(0.8, 0.5)) == 0.0


def test_optimal_cost_fractions_hand_built():
    preference = np.array(
        [
            [0.5, 0.8, 0.9],
            [0.2, 0.5, 0.7],
            [0.1, 0.3, 0.5],
        ],
        dtype=float,
    )
    costs = np.array([1.0, 4.0, 2.0], dtype=float)
    winner = 0
    arms = (0, 1, 2)

    alpha_expected = {m: 0.0 for m in arms}
    gamma_expected = {m: [] for m in arms}
    for m in arms:
        if m == winner:
            continue
        scores = []
        for n in arms:
            if n == m:
                continue
            p_mn = float(preference[m, n])
            gap = 0.0 if p_mn >= 0.5 else _kl(p_mn, 0.5)
            scores.append((n, gap / (float(costs[m]) + float(costs[n]))))
        best = max(score for _, score in scores)
        alpha_expected[m] = best
        gamma_expected[m] = [n for n, score in scores if score == best]

    inv_sum = sum(1.0 / alpha_expected[n] for n in arms if n != winner)
    pair_mass = {(i, j): 0.0 for i in arms for j in arms if i < j}
    for m in arms:
        if m == winner:
            continue
        share = (1.0 / alpha_expected[m]) / inv_sum / len(gamma_expected[m])
        for j in gamma_expected[m]:
            a, b = (m, j) if m < j else (j, m)
            pair_mass[(a, b)] += share

    raw = {
        pair: mass / (float(costs[pair[0]]) + float(costs[pair[1]]))
        for pair, mass in pair_mass.items()
    }
    denom = sum(raw.values())
    pull_expected = {pair: value / denom for pair, value in raw.items()}

    got = optimal_cost_fractions(preference, costs)
    assert got.winner == winner
    assert got.alpha_m[winner] == 0.0
    for m in arms:
        if m == winner:
            assert got.gamma[m] == ()
            continue
        assert abs(got.alpha_m[m] - alpha_expected[m]) < 1e-9
        assert set(got.gamma[m]) == set(gamma_expected[m])

    upper = [(i, j) for i in arms for j in arms if i < j]
    w_sum = 0.0
    pull_sum = 0.0
    for i, j in upper:
        assert got.W[i, j] >= -1e-15
        assert got.pull_alpha[i, j] >= -1e-15
        assert abs(got.W[i, j] - pair_mass[(i, j)]) < 1e-9
        assert abs(got.pull_alpha[i, j] - pull_expected[(i, j)]) < 1e-9
        assert abs(got.W[i, j] - got.W[j, i]) < 1e-12
        w_sum += got.W[i, j]
        pull_sum += got.pull_alpha[i, j]
    assert abs(w_sum - 1.0) < 1e-9
    assert abs(pull_sum - 1.0) < 1e-9
    assert any(pull_expected[pair] == 0.0 for pair in upper)
    assert any(got.pull_alpha[i, j] == 0.0 for i, j in upper)


def _easy_instance():
    preference = np.array(
        [
            [0.5, 0.95, 0.95],
            [0.05, 0.5, 0.6],
            [0.05, 0.4, 0.5],
        ],
        dtype=float,
    )
    costs = np.array([1.0, 20.0, 2.0], dtype=float)
    return preference, costs


def _uniform_pair_sampler(preference, costs, delta, seed, max_pulls=20000):
    """Uniform pairs after one pull of each pair, same Z and beta stop."""
    rng = np.random.default_rng(seed)
    p = np.asarray(preference, dtype=float)
    c = np.asarray(costs, dtype=float)
    L = p.shape[0]
    pair_list = [(i, j) for i in range(L) for j in range(i + 1, L)]
    wins = np.zeros((L, L), dtype=np.int64)
    counts = np.zeros((L, L), dtype=np.int64)
    total_cost = 0.0

    def pull(i, j):
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
    stopped = False
    winner = None
    while t_total < max_pulls:
        i, j = pair_list[int(rng.integers(0, len(pair_list)))]
        pull(i, j)
        t_total += 1
        Z = glr(wins, counts)
        threshold = beta(t_total, delta, L)
        for arm in range(L):
            if all(Z[arm, other] > threshold for other in range(L) if other != arm):
                winner = arm
                stopped = True
                break
        if stopped:
            break
    return winner, t_total, total_cost, stopped


def test_dctas_returns_winner_before_cap():
    preference, costs = _easy_instance()
    delta = 0.2
    for seed in (0, 1, 2, 3, 4):
        result = dctas(preference, costs, delta=delta, seed=seed, max_pulls=20000)
        assert result.stopped, f"seed {seed} hit the pull cap"
        assert result.duels < 20000
        assert result.winner == 0


def test_dctas_cheaper_than_uniform_pairs():
    preference, costs = _easy_instance()
    delta = 0.2
    seeds = (0, 1, 2, 3, 4, 5, 6, 7)
    dctas_costs = []
    uniform_costs = []
    for seed in seeds:
        result = dctas(preference, costs, delta=delta, seed=seed, max_pulls=20000)
        assert result.stopped and result.winner == 0
        other = _uniform_pair_sampler(preference, costs, delta, seed, max_pulls=20000)
        winner, duels, total_cost, stopped = other
        assert stopped, f"uniform seed {seed} did not stop"
        assert duels < 20000
        assert winner == 0
        dctas_costs.append(result.total_cost)
        uniform_costs.append(total_cost)
    assert sum(dctas_costs) / len(dctas_costs) < sum(uniform_costs) / len(uniform_costs)
