"""Runnable demo on synthetic tokens.

python -m coverpruner.demo --tokens 576 --budget 64

Real visual tokens are clustered: large redundant regions (sky, wall) plus a few
small distinct ones (text patches, chart marks). The generator here mirrors that,
with cluster sizes drawn from a power law so most tokens sit in a handful of big
groups. There is no VLM in the loop, so the numbers below are values of the RCM
objective F(S) (Eq. 5), not benchmark accuracy.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from coverpruner.coverage import calibrated_utilities, total_coverage
from coverpruner.relevance import demand_weights
from coverpruner.select import lazy_greedy_select


def synthetic_tokens(key, n, d=64, clusters=12, spread=0.35):
    """Draw n tokens from `clusters` centers with uneven populations, plus noise."""
    k_center, k_assign, k_noise, k_rel = jax.random.split(key, 4)
    centers = jax.random.normal(k_center, (clusters, d))
    centers = centers / jnp.linalg.norm(centers, axis=1, keepdims=True)

    weights = 1.0 / jnp.arange(1, clusters + 1) ** 1.2
    assign = jax.random.categorical(k_assign, jnp.log(weights), shape=(n,))
    tokens = centers[assign] + spread * jax.random.normal(k_noise, (n, d))

    # Stand-in for the layer-0 probe of Eq. 4: the instruction cares about one small
    # cluster, so demand concentrates on tokens that a plain coverage run may skip.
    target = clusters - 1
    r = jnp.where(assign == target, 1.0, 0.0)
    r = r + 0.15 * jax.random.uniform(k_rel, (n,))
    return tokens, assign, r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coverpruner.demo")
    ap.add_argument("--tokens", type=int, default=576, help="n, visual tokens")
    ap.add_argument("--budget", type=int, default=64, help="m, retained tokens")
    ap.add_argument("--dim", type=int, default=64, help="d, projector width")
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.3, help="Eq. 4 temperature")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=20, help="random baseline draws")
    args = ap.parse_args(argv)

    key = jax.random.key(args.seed)
    V, assign, r = synthetic_tokens(key, args.tokens, args.dim, args.clusters)
    A = calibrated_utilities(V)
    w = demand_weights(r, args.tau)

    sel = lazy_greedy_select(A, w, args.budget)
    f_cover = float(sel.coverage[-1])

    rng = np.random.default_rng(args.seed)
    randoms = [
        float(total_coverage(A, w, rng.choice(args.tokens, args.budget, replace=False)))
        for _ in range(args.trials)
    ]
    topk = jnp.argsort(-r)[: args.budget]
    f_topk = float(total_coverage(A, w, topk))
    f_all = float(total_coverage(A, w, jnp.arange(args.tokens)))

    print(f"n = {args.tokens} tokens, d = {args.dim}, {args.clusters} clusters")
    print(f"budget m = {args.budget}, tau = {args.tau}")
    print()
    print(f"{'selector':<26}{'F(S)':>10}{'% of F(all)':>14}")
    for name, value in [
        ("CoverPruner (greedy)", f_cover),
        (f"random (mean of {args.trials})", float(np.mean(randoms))),
        ("top-k by attention", f_topk),
        ("all tokens kept", f_all),
    ]:
        print(f"{name:<26}{value:>10.3f}{100 * value / f_all:>13.1f}%")
    print()

    kept = np.asarray(assign)[np.asarray(sel.indices)]
    hit = len(set(kept.tolist()))
    print(f"clusters represented: {hit}/{args.clusters}")
    print(f"first 8 picks (in order): {np.asarray(sel.order)[:8].tolist()}")


if __name__ == "__main__":
    main()
