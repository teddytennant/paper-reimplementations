"""Train the tiny encoder warm start and write a conformal certificate."""

import json

import jax
import jax.numpy as jnp

from robust_rl import conformal_quantile, q_star, train_warm_start, warm_start_q

R = 0.2
GAMMA = 0.9
MAZE = 5
TRAIN_GOALS = [0, 4, 6, 12, 18, 20, 24]
# N=12, delta=0.1 -> rank = ceil(0.9*13) = 12, the largest finite score.
CAL_GOALS = [1, 2, 3, 5, 8, 10, 11, 14, 16, 17, 22, 23]
HOLDOUT = 15


def inf_error(params, goal):
    target = q_star(goal, R, GAMMA)
    pred = warm_start_q(goal, R, params)
    return float(jnp.max(jnp.abs(pred - target))), float(jnp.max(jnp.abs(target)))


def main():
    params = train_warm_start(TRAIN_GOALS, R, GAMMA, steps=200, lr=0.15)
    warm, zero = inf_error(params, HOLDOUT)
    cal = jnp.asarray([inf_error(params, g)[0] for g in CAL_GOALS])
    bound = float(conformal_quantile(cal, 0.1))
    results = {
        "warm_start_error": warm,
        "zero_init_error": zero,
        "conformal_bound": bound,
        "heldout_error": warm,
        "R": R,
        "gamma": GAMMA,
        "maze_size": MAZE,
        "calibration_scores": [float(x) for x in cal],
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))
    finite = bound == bound and bound != float("inf")
    if not (warm < zero and finite and bound >= warm):
        raise SystemExit(
            f"certificate failed: warm={warm} zero={zero} bound={bound}"
        )


if __name__ == "__main__":
    main()
