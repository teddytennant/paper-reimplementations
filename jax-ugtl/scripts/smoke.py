"""A larger UGTL run on the negated Branin task, for checking an accelerator.

    python scripts/smoke.py

Prints the JAX backend, the offline best, and what the pipeline returned. It runs
on a CPU too, just slower.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import numpy as np

from ugtl.diagnostics import cumulative_regret, smoothness
from ugtl.pipeline import UGTLConfig, run_ugtl
from ugtl.tasks import branin_task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=25_000)
    args = ap.parse_args(argv)

    print(f"jax {jax.__version__} on {jax.default_backend()}: {jax.devices()}")

    config = UGTLConfig(
        horizon=48,
        n_traj=600,
        ensemble_steps=3000,
        ensemble_width=256,
        n_ensemble=10,
        T_diff=200,
        diffusion_steps=10_000,
        diffusion_width=256,
        H_ctx=24,
        n_gen=512,
        M_pool=2048,
        Q=32,
    )

    t0 = time.time()
    k_task, k_run = jax.random.split(jax.random.PRNGKey(args.seed))
    task = branin_task(k_task, n_samples=args.samples, keep_fraction=0.6)
    print(f"offline dataset: {len(task.X)} designs, best normalized score {task.best_offline:.4f}")

    best = {}
    for name in ("ugtl", "random"):
        t1 = time.time()
        result = run_ugtl(k_run, task, config, name)
        scores = task.evaluate(result.candidates)
        traj = result.trajectories
        best[name] = scores.max()
        print(
            f"{name:6s} best {scores.max():.4f}  median {np.median(scores):.4f}  "
            f"regret {cumulative_regret(traj.scores).mean():6.2f}  "
            f"smoothness {smoothness(traj.designs, task.lo, task.hi).mean():9.4f}  "
            f"[{time.time() - t1:.1f}s]"
        )

    print(f"UGTL improves on the offline best by {best['ugtl'] - task.best_offline:+.4f}")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
