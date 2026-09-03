"""Run a toy MFSPNet search and print what it found.

    python -m mfspnet

This is Alg. 3 on synthetic 8x8 images with a small search space. Most of the
wall clock is XLA compiling one graph per distinct architecture, so the cost
tracks the number of distinct candidates, not the number of gradient steps. It
is not the paper's search; that is 20 particles over 55 generations on
downscaled CIFAR-10 (Table 3).
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from .data import synthetic_task
from .encoding import SearchSpace
from .model import StackConfig, count_parameters, init_network
from .search import SearchConfig, random_search, search


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4, help="validation cut-off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack", type=int, default=3, help="blocks to stack at the end")
    parser.add_argument("--compare-random", action="store_true")
    args = parser.parse_args(argv)

    print(f"jax {jax.__version__} on {jax.default_backend()}, devices {jax.devices()}")

    space = SearchSpace(l_max=4, r_lower=2, r_upper=4)
    data = synthetic_task(jax.random.key(0), n_train=128, n_val=64)
    cfg = SearchConfig(
        n_particles=args.particles,
        n_generations=args.generations,
        cutoff_epochs=args.epochs,
        batch_size=64,
        stem_channels=6,
        n_blocks=1,
    )

    started = time.time()
    result = search(jax.random.key(1), data, space, cfg, seed=args.seed)
    elapsed = time.time() - started
    print(f"evolved block growth rates {result.growth_rates}")
    print(f"VLE-EMA {result.fitness:.4f} after {result.n_trained} trained candidates")
    print(f"convergence {np.round(result.history, 4).tolist()}")
    print(f"search took {elapsed:.1f}s")

    if args.compare_random:
        sampled = random_search(
            jax.random.key(1), data, space, result.n_trained, cfg, seed=args.seed
        )
        print(f"random search over the same budget: {sampled.fitness:.4f} {sampled.growth_rates}")

    stack_cfg = StackConfig(n_classes=data.n_classes, stem_channels=6, n_blocks=args.stack)
    params = init_network(jax.random.key(2), result.growth_rates, data.in_channels, stack_cfg)
    print(
        f"stacked x{args.stack} with Eq. 11 aggregation: "
        f"{count_parameters(params)} parameters"
    )


if __name__ == "__main__":
    main()
