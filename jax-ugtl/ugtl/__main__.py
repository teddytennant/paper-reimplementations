"""Run UGTL on the negated Branin task and print what came back.

    python -m ugtl
"""

from __future__ import annotations

import argparse

import jax
import numpy as np

from ugtl.diagnostics import cumulative_regret, smoothness
from ugtl.pipeline import UGTLConfig, run_ugtl
from ugtl.tasks import branin_task


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="UGTL on the negated Branin function")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--keep", type=float, default=0.6, help="fraction of the sample kept as offline data")
    ap.add_argument("--candidates", type=int, default=16, help="the candidate budget Q")
    ap.add_argument("--ablation", action="store_true", help="also run the random-construction ablation")
    args = ap.parse_args(argv)

    k_task, k_run = jax.random.split(jax.random.PRNGKey(args.seed))
    task = branin_task(k_task, n_samples=args.samples, keep_fraction=args.keep)
    config = UGTLConfig(Q=args.candidates)

    print(f"offline dataset: {len(task.X)} designs, best normalized score {task.best_offline:.4f}")
    for name in ("ugtl", "random") if args.ablation else ("ugtl",):
        result = run_ugtl(k_run, task, config, name)
        scores = task.evaluate(result.candidates)
        traj = result.trajectories
        print(
            f"{name:6s} best {scores.max():.4f}  median {np.median(scores):.4f}  "
            f"regret {cumulative_regret(traj.scores).mean():6.2f}  "
            f"smoothness {smoothness(traj.designs, task.lo, task.hi).mean():8.4f}"
        )


if __name__ == "__main__":
    main()
