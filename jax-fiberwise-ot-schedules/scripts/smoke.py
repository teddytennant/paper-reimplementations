"""One-shot 1D mixture smoke for fiberwise OT schedules."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from fiberwise_ot.train import one_shot_retrain


def main():
    key = jax.random.PRNGKey(0)
    out = one_shot_retrain(
        key,
        family="ddpm",
        baseline_steps=20,
        retrain_steps=20,
        batch=64,
        n_intervals=16,
        n_risk_samples=128,
    )
    print("baseline_loss", float(out["baseline_loss"]))
    print("retrain_loss", float(out["retrain_loss"]))
    print("w_sum", float(out["w"].sum()))
    print("r_min", float(out["r"].min()), "r_max", float(out["r"].max()))


if __name__ == "__main__":
    main()
