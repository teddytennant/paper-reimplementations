"""Reproduce the synthetic validation of Section 4.1 / Appx. C.1.

Runs Oracle MH, Pref-MH and the plug-in baseline on the 241-state target with
N in {1, 2, 4}, and reports the cumulative empirical TV distance to pi at a few
checkpoints. Chains start at k = 0 and no burn-in is discarded, matching the
paper, so the early numbers are dominated by the walk away from the
initialization. Pref-MH keeps dropping; the plug-in flattens out.

Run with ``python -m pref_mh.demo``.
"""

import argparse

import jax
import numpy as np

from .x64 import enable

enable()

from .finite import sampler_ops, total_variation  # noqa: E402
from .sampler import oracle_mh_chain, plugin_mh_chain, pref_mh_chain  # noqa: E402
from .synthetic import N_STATES, synthetic_model  # noqa: E402


def cumulative_tv(states, pi, checkpoints):
    """TV(hat-pi_t, pi) at each t, with hat-pi_t the running histogram."""
    counts = np.zeros(len(pi))
    out = []
    prev = 0
    s = np.asarray(states)
    for t in checkpoints:
        idx, cnt = np.unique(s[prev:t], return_counts=True)
        counts[idx] += cnt
        prev = t
        out.append(total_variation(counts / t, pi))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = synthetic_model()
    ops = sampler_ops(model)
    pi = np.asarray(model.target())
    key = jax.random.PRNGKey(args.seed)
    checkpoints = [args.steps // 10, args.steps // 2, args.steps]

    rows = []
    states, acc = oracle_mh_chain(
        key, 0, propose=ops["propose"], log_q=ops["log_q"],
        log_target=ops["log_target"], n_steps=args.steps,
    )
    rows.append(("Oracle MH", "-", float(acc.mean()),
                 cumulative_tv(states, pi, checkpoints)))

    for n in (1, 2, 4):
        states, acc = pref_mh_chain(
            key, 0, propose=ops["propose"], log_q=ops["log_q"],
            log_p0=ops["log_p0"], judges=ops["judges"], n_votes=n,
            n_steps=args.steps,
        )
        rows.append(("Pref-MH", n, float(acc.mean()),
                     cumulative_tv(states, pi, checkpoints)))

    for n in (1, 2, 4):
        states, acc = plugin_mh_chain(
            key, 0, propose=ops["propose"], log_q=ops["log_q"],
            log_p0=ops["log_p0"], judge=ops["judges"][0], n_votes=n,
            n_steps=args.steps,
        )
        rows.append(("Plug-in", n, float(acc.mean()),
                     cumulative_tv(states, pi, checkpoints)))

    print(f"{N_STATES}-state synthetic target, {args.steps} steps, seed "
          f"{args.seed}, no burn-in discarded")
    head = "  ".join(f"TV@{t}" for t in checkpoints)
    print(f"{'method':<10}{'N':>3}{'accept':>9}   {head}")
    for name, n, a, tvs in rows:
        body = "  ".join(f"{v:8.4f}" for v in tvs)
        print(f"{name:<10}{str(n):>3}{a:9.3f}   {body}")


if __name__ == "__main__":
    main()
