"""Selector tests: Algorithm 1 / Eq. 6 greedy, lazy greedy, demand, and the probe."""

import numpy as np
import reference as ref

from coverpruner import (
    calibrated_utilities,
    demand_weights,
    greedy_select,
    layer0_attention_probe,
    lazy_greedy_select,
    relevance_from_probe,
    select_tokens,
    total_coverage,
)


def numpy_greedy(A, w, budget):
    """Independent greedy: k* = argmax_k sum_i w_i max(0, A_ik - c_i), ties to lowest index."""
    A = np.asarray(A, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = A.shape[0]
    c = np.zeros(n, dtype=np.float64)
    chosen = np.zeros(n, dtype=bool)
    order = []
    for _ in range(budget):
        gains = w @ np.maximum(0.0, A - c[:, None])
        gains[chosen] = -np.inf
        k = int(np.argmax(gains))
        chosen[k] = True
        order.append(k)
        c = np.maximum(c, A[:, k])
    return np.asarray(order, dtype=np.int32)


def _instances():
    for seed in range(8):
        n = 16 + seed
        m = 4 + (seed % 3)
        A, w, V, _ = ref.random_instance(seed, n=n, d=12, clusters=4)
        yield seed, A, w, V, m


def test_greedy_matches_numpy_reference():
    for seed, A, w, _, m in _instances():
        got = np.asarray(greedy_select(A, w, m).order)
        expect = numpy_greedy(A, w, m)
        assert np.array_equal(got, expect), seed


def test_lazy_greedy_matches_greedy_order():
    for seed, A, w, _, m in _instances():
        g = np.asarray(greedy_select(A, w, m).order)
        lazy = np.asarray(lazy_greedy_select(A, w, m).order)
        assert np.array_equal(g, lazy), seed


def test_greedy_meets_one_minus_one_over_e():
    """Nemhauser et al.: F(greedy) >= (1 - 1/e) F(OPT) for monotone submodular F."""
    m = 3
    ratio = 1.0 - 1.0 / np.e
    for seed in range(6):
        n = 8 + seed % 3  # n in {8, 9, 10}
        A, w, _, _ = ref.random_instance(seed, n=n, d=6, clusters=3)
        opt, _ = ref.brute_force(A, w, m)
        sel = greedy_select(A, w, m)
        f_g = float(total_coverage(A, w, sel.indices))
        assert f_g >= ratio * opt - 1e-10, (seed, n, f_g, opt)


def test_demand_weights_tau_zero_constant_and_argmax():
    r = np.array([0.2, 1.4, 0.7, 0.0, 0.9])
    w0 = np.asarray(demand_weights(r, tau=0.0))
    assert np.allclose(w0, 1.0)

    w_const = np.asarray(demand_weights(np.full(7, 3.5), tau=0.3))
    assert np.allclose(w_const, w_const[0])
    assert np.allclose(w_const, 1.0)

    w = np.asarray(demand_weights(r, tau=0.3))
    assert int(np.argmax(w)) == int(np.argmax(r))
    assert w[np.argmax(r)] == w.max()


def test_layer0_probe_softmax_and_relevance_average():
    rng = np.random.default_rng(11)
    t_q, n, d, h, d_head = 5, 9, 16, 4, 4
    h_instruction = rng.normal(size=(t_q, d))
    h_visual = rng.normal(size=(n, d))
    w_q = rng.normal(size=(d, h * d_head))
    w_k = rng.normal(size=(d, h * d_head))

    alpha = np.asarray(
        layer0_attention_probe(h_instruction, h_visual, w_q, w_k, h)
    )
    assert alpha.shape == (h, t_q, n)
    assert np.allclose(alpha.sum(axis=-1), 1.0)

    r = np.asarray(relevance_from_probe(alpha))
    expect = alpha.mean(axis=(0, 1))
    assert r.shape == (n,)
    assert np.allclose(r, expect)


def test_coverage_beats_random_and_topk_on_redundant_clusters():
    """Coverage spends budget on distinct clusters; scoring keeps redundant high-r tokens.

    Seed 4, n=30, 5 clusters of roughly equal size, r concentrated on cluster 0,
    unweighted F (tau=0). Verified that greedy F strictly exceeds the mean of 20
    random subsets and F of top-k by raw relevance.
    """
    rng = np.random.default_rng(4)
    n, d, clusters, m, trials = 30, 8, 5, 5, 20
    C = rng.normal(size=(clusters, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    grp = np.repeat(np.arange(clusters), n // clusters)
    V = C[grp] + 0.12 * rng.normal(size=(n, d))
    r = np.where(grp == 0, 1.0, 0.08)
    A = np.asarray(calibrated_utilities(V))
    w = np.ones(n)

    sel = greedy_select(A, w, m)
    f_g = float(total_coverage(A, w, sel.indices))

    randoms = [
        float(total_coverage(A, w, rng.choice(n, m, replace=False)))
        for _ in range(trials)
    ]
    topk = np.argsort(-r)[:m]
    f_topk = float(total_coverage(A, w, topk))

    assert f_g > float(np.mean(randoms))
    assert f_g > f_topk


def test_select_tokens_sorted_indices_and_greedy_order():
    A, w, V, _ = ref.random_instance(2, n=20, d=10, clusters=4)
    r = np.log(np.asarray(w)) / 0.3
    sel = select_tokens(V, budget=6, r=r, tau=0.3, lazy=False)
    order = np.asarray(sel.order)
    indices = np.asarray(sel.indices)
    assert np.array_equal(indices, np.sort(order))
    expect = np.asarray(greedy_select(calibrated_utilities(V), demand_weights(r, 0.3), 6).order)
    assert np.array_equal(order, expect)


def test_empty_budget_returns_empty_indices():
    A, w, _, _ = ref.random_instance(0, n=12)
    sel = greedy_select(A, w, 0)
    assert np.asarray(sel.indices).shape == (0,)
    assert np.asarray(sel.order).shape == (0,)
    assert np.asarray(sel.coverage).shape == (0,)
