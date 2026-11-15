"""Smoke entry: python -m hbds.finetune --steps 40"""

import argparse

from hbds.path import run_finetune


def main(argv=None):
    parser = argparse.ArgumentParser(description="Path-regularized score fine-tune smoke")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lambda-path", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_finetune(
        steps=args.steps,
        seed=args.seed,
        lambda_path=args.lambda_path,
    )
    losses = result["losses"]
    initial = float(losses[0])
    final = float(losses[-1])
    if not (initial == initial and final == final):
        raise SystemExit("fine-tune loss was not finite")
    print(
        f"steps={args.steps} n_params={result['n_params']} "
        f"initial_loss={initial:.6f} final_loss={final:.6f}"
    )
    return result


if __name__ == "__main__":
    main()
