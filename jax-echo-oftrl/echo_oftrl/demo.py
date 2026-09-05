"""Print individual regret against the horizon for ECHO-OFTRL and two controls.

    python -m echo_oftrl.demo --horizon 2000

Controls, both reached by changing only the cascade parameters:
  last  depth 1, rho 0: ghat^(t) = g^(t-1), one-step-recency optimism.
  none  depth 1, rho 1: ghat = 0, so plain lifted FTRL with no optimism.
"""

from __future__ import annotations

import argparse

import numpy as np

from .dynamics import DEFAULT_ETA, run_game
from .games import biased_matching_pennies, duality_gap, random_game

VARIANTS = {"echo": {}, "last": dict(depth=1, rho=0.0), "none": dict(depth=1, rho=1.0)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="echo_oftrl.demo")
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--actions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    horizons = [h for h in (args.horizon // 8, args.horizon // 4, args.horizon // 2, args.horizon) if h > 0]

    zs = biased_matching_pennies()
    print(f"eta = {args.eta}")
    print("\nbiased matching pennies, equilibrium (1/3, 2/3), value 1/3")
    print(f"{'T':>7}  {'max Reg_i':>10}  {'sum Reg_i':>10}  {'gap(avg)':>10}")
    for h in horizons:
        tr = run_game(zs, h, eta=args.eta)
        reg = np.asarray(tr.final_regret)
        print(f"{h:>7}  {reg.max():>10.4f}  {reg.sum():>10.4f}  {float(duality_gap(zs, tr.mean_strategy)):>10.6f}")

    game = random_game(args.players, args.actions, args.seed)
    print(f"\nrandom {args.players}-player {args.actions}-action game, seed {args.seed}")
    print(f"{'T':>7}  " + "  ".join(f"{k:>10}" for k in VARIANTS))
    for h in horizons:
        cells = []
        for kwargs in VARIANTS.values():
            tr = run_game(game, h, eta=args.eta, **kwargs)
            cells.append(f"{float(tr.final_regret.max()):>10.4f}")
        print(f"{h:>7}  " + "  ".join(cells))
    print("\ncolumns are max_i Reg_i(T); horizon-independence is the flat column")


if __name__ == "__main__":
    main()
