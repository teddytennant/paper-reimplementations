"""Reproduce the finite-horizon experiment of Section 6.1.

Algorithm 1 with T = 1000 and eta_0 = 0.5, then Algorithm 2 with K = 300
blocks of B = 200 episodes, zeta = 0.05 and eta_0 = 0.1, on the four games of
Appendix F. The reported number is ``Gap_N``, the max over players of the
exact general-policy CCE gap of the first ``N`` profiles.

The paper uses B = 500 over K = 1000 blocks, which is 500,000 episodes per game
per seed. This script uses a smaller budget so it finishes in a couple of
minutes on a laptop.
"""

from __future__ import annotations

import argparse

import jax

from . import algorithm, gap, games


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--block-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x64", action="store_true", default=True)
    args = parser.parse_args(argv)

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    checkpoints = [1, 2, 4, 8]
    for name, game in games.benchmark_games().items():
        traj = algorithm.run_full_feedback(game, args.episodes, eta0=0.5)
        row = " ".join(
            f"N={args.episodes // c:<5d} {gap.cce_gap(game, tuple(a[: args.episodes // c] for a in traj)):.5f}"
            for c in reversed(checkpoints)
        )
        print(f"full-feedback    {name:26s} {row}")

    for name, game in games.benchmark_games().items():
        traj = algorithm.run_partial_feedback(
            game,
            args.blocks,
            args.block_length,
            jax.random.key(args.seed),
            eta0=0.1,
            zeta=0.05,
        )
        row = " ".join(
            f"K={args.blocks // c:<5d} {gap.cce_gap(game, tuple(a[: args.blocks // c] for a in traj)):.5f}"
            for c in reversed(checkpoints)
        )
        print(f"partial-feedback {name:26s} {row}")


if __name__ == "__main__":
    main()
