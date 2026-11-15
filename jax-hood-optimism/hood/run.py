"""Play a game under HOOD and print max individual regret against horizon.

    python -m hood.run --players 3 --actions 4 --rounds 5000
"""

import argparse

import jax

from hood.algorithms import run_hood, run_optftrl
from hood.games import coordination_game, matching_pennies, random_game, rock_paper_scissors
from hood.regret import (
    cce_gap,
    empirical_distribution,
    individual_regret,
    max_step_size,
    regret_bound,
)


def build_game(name, num_players, num_actions, seed):
    if name == "random":
        return random_game(jax.random.key(seed), num_players, num_actions)
    if name == "matching-pennies":
        return matching_pennies()
    if name == "rps":
        return rock_paper_scissors()
    if name == "coordination":
        return coordination_game(num_players, num_actions)
    raise ValueError(f"unknown game {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--game",
        default="random",
        choices=["random", "matching-pennies", "rps", "coordination"],
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="step size, default is the largest Theorem 1 allows, 1/(60(N+1)^3)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also run OMWU at the same step size for comparison",
    )
    args = parser.parse_args(argv)

    game = build_game(args.game, args.players, args.actions, args.seed)
    n, k = game.num_players, game.num_actions
    eta = args.eta if args.eta is not None else max_step_size(n)
    bound = regret_bound(k, eta)

    traj = run_hood(game, eta, args.rounds)
    regret = individual_regret(traj.g)

    print(f"game={args.game} N={n} K={k} eta={eta:.6g} T={args.rounds}")
    print(f"Theorem 1 bound 3L/eta = {bound:.4g}")
    print(f"{'T':>8}  {'max_i Reg_i(T)':>15}  {'CCE gap':>10}  {'3L/(eta T)':>12}")
    checkpoints = [t for t in (1, 10, 100, 1000, 10000, 100000) if t <= args.rounds]
    if args.rounds not in checkpoints:
        checkpoints.append(args.rounds)
    for t in checkpoints:
        worst = float(regret[t - 1].max())
        gap = float(cce_gap(game, empirical_distribution(traj.x[:t])).max())
        print(f"{t:>8}  {worst:>15.6f}  {gap:>10.6f}  {bound / t:>12.4g}")
    print(f"sup over T of max_i Reg_i(T) = {float(regret.max()):.6f}")

    if args.baseline:
        base = run_optftrl(game, eta, args.rounds)
        base_regret = individual_regret(base.g)
        print("\nOMWU baseline, same step size")
        print(f"{'T':>8}  {'max_i Reg_i(T)':>15}")
        for t in checkpoints:
            print(f"{t:>8}  {float(base_regret[t - 1].max()):>15.6f}")


if __name__ == "__main__":
    main()
