#!/usr/bin/env python3
"""GPU-ready smoke: synthetic irregular trajectories, tiny train, K completions."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from jax_flowatc.baselines import constant_velocity_ade
from jax_flowatc.cfm import sample_completions
from jax_flowatc.config import tiny_config
from jax_flowatc.data import (
    apply_zscore,
    deltas_and_elapsed,
    invert_zscore,
    synthetic_with_turn,
    zscore_stats,
)
from jax_flowatc.metrics import min_ade_at_k
from jax_flowatc.model import init_params
from jax_flowatc.train import adamw_init, make_train_step


def _finite(x) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())


def main() -> int:
    cfg = tiny_config(dropout=0.0)
    key = jax.random.PRNGKey(0)
    k_data, k_init, k_train, k_samp = jax.random.split(key, 4)

    states, times = synthetic_with_turn(
        k_data, batch=16, t_obs=cfg.t_obs, t_fut=cfg.t_fut, turn_rad=0.7
    )
    stats = zscore_stats(states)
    z = apply_zscore(states, stats)
    dt, t_rel = deltas_and_elapsed(times)
    x_obs_z = z[:, : cfg.t_obs]
    x_fut_z = z[:, cfg.t_obs :]

    params = init_params(k_init, cfg)
    opt = adamw_init(params)
    step = make_train_step(cfg, lr=3e-3, wd=0.0)

    n_train = 12
    keys = jax.random.split(k_train, n_train)
    loss = jnp.array(jnp.nan)
    for i in range(n_train):
        params, opt, loss, grads = step(params, opt, x_obs_z, x_fut_z, dt, t_rel, keys[i])
        if not _finite(loss) or not all(_finite(g) for g in jax.tree.leaves(grads)):
            print("non-finite loss or grads", flush=True)
            return 1

    # one example for sampling / metrics
    i = 0
    completions_z = sample_completions(
        params,
        cfg,
        x_obs_z[i],
        dt[i],
        t_rel[i],
        k_samp,
        n_steps=8,
        k=4,
    )
    completions = invert_zscore(completions_z, stats)
    gt = states[i, cfg.t_obs :]
    ade_k = min_ade_at_k(completions, gt)
    cv_ade = constant_velocity_ade(
        states[i, : cfg.t_obs], gt, times[i, : cfg.t_obs], times[i, cfg.t_obs :]
    )

    print(f"device={jax.default_backend()} loss={float(loss):.6f}")
    print(f"minADE@4={float(ade_k):.6f}  constant-velocity ADE={float(cv_ade):.6f}")

    if not (_finite(loss) and _finite(ade_k) and _finite(cv_ade) and _finite(completions)):
        print("non-finite metric", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
