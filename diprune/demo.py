"""Walk through the N=5 score disagreement and a 4-token bound comparison.

No model is trained. No benchmark number is reported. The bound weights are
the theorem, not the DIPrune inference score.
"""

import numpy as np

from diprune import (
    alignment_score,
    fuse_scores,
    inner_bound,
    relaxed_bound,
    select_topk,
    semantic_evolution,
    token_weights,
    topk_mask,
)


def _fmt(values):
    return " ".join(f"{float(value):.4f}" for value in values)


def _kept_name(names, scores):
    index = int(select_topk(scores, 1)[0])
    return names[index]


def _worst_mask(weights, k):
    weights = np.asarray(weights)
    index = np.arange(weights.shape[0])
    order = np.lexsort((index, weights))
    mask = np.zeros(weights.shape[0], dtype=np.int32)
    mask[order[:k]] = 1
    return mask


def main():
    names = ("static", "rising", "mid", "low", "drop")
    ranks_prev = np.array([1, 5, 2, 3, 4])
    ranks_curr = np.array([1, 2, 3, 4, 5])
    m_max = 2
    s_align = alignment_score(ranks_curr)
    s_evol = semantic_evolution(ranks_prev, ranks_curr, m_max=m_max)
    fused = fuse_scores(s_align, s_evol, alpha=0.6, beta=0.4)
    only_align = fuse_scores(s_align, s_evol, alpha=1.0, beta=0.0)
    only_evol = fuse_scores(s_align, s_evol, alpha=0.0, beta=1.0)

    print("tokens", " ".join(names))
    print("ranks prev", " ".join(str(int(rank)) for rank in ranks_prev))
    print("ranks curr", " ".join(str(int(rank)) for rank in ranks_curr))
    print("S_align", _fmt(s_align))
    print("S_evol", _fmt(s_evol))
    print("fused alpha=0.6 beta=0.4", _fmt(fused))
    print("alpha=1 beta=0 keeps", _kept_name(names, only_align))
    print("alpha=0 beta=1 keeps", _kept_name(names, only_evol))
    print("alpha=0.6 beta=0.4 keeps", _kept_name(names, fused))

    rng = np.random.default_rng(0)
    v = rng.normal(size=(4, 3))
    g = rng.normal(size=(4, 3))
    mu = 0.5
    l_f = 1.0
    k = 2
    weights = token_weights(v, g, mu, l_f)
    mask_top = np.asarray(topk_mask(weights, k))
    mask_worst = _worst_mask(weights, k)
    b_inner_top = inner_bound(v, g, mask_top, l_f)
    b_relaxed_top = relaxed_bound(v, g, mask_top, mu, l_f)
    b_inner_worst = inner_bound(v, g, mask_worst, l_f)
    b_relaxed_worst = relaxed_bound(v, g, mask_worst, mu, l_f)

    print("bound tokens", 4, "keep", k)
    print("w", _fmt(weights))
    print("top-k mask", " ".join(str(int(bit)) for bit in mask_top))
    print("worst mask", " ".join(str(int(bit)) for bit in mask_worst))
    print("B_inner top-k", f"{float(b_inner_top):.6f}")
    print("B_relaxed top-k", f"{float(b_relaxed_top):.6f}")
    print("B_inner worst", f"{float(b_inner_worst):.6f}")
    print("B_relaxed worst", f"{float(b_relaxed_worst):.6f}")
    print(
        "relaxed bound is at least the inner bound, and the top-k mask "
        "beats the worst mask of the same budget"
    )
    if float(b_relaxed_top) + 1e-5 < float(b_inner_top):
        raise SystemExit("relaxed bound fell below the inner bound")
    if float(b_relaxed_worst) + 1e-5 < float(b_inner_worst):
        raise SystemExit("relaxed bound fell below the inner bound")
    if not (float(b_relaxed_top) < float(b_relaxed_worst)):
        raise SystemExit("top-k did not beat the worst mask")


if __name__ == "__main__":
    main()
