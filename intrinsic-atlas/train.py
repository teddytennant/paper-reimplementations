"""Train a tiny 2D DDPM and run posterior-mean / DDIM / stochastic samplers."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from intrinsic_atlas.ddpm import adam_init, init_model, make_learned_x0_fn, train_step
from intrinsic_atlas.gaussian import FIG4_LAMBDAS, make_gaussian_x0_fn
from intrinsic_atlas.sampler import (
    sample_ddim,
    sample_posterior_mean,
    sample_stochastic,
    spread_offset,
)
from intrinsic_atlas.schedules import make_vp_schedule


T = 200
STEPS = 2500
BATCH = 256
LR = 1e-3
N_SEEDS = 16
MU = jnp.array([0.25, -0.15])
LAM = jnp.array(FIG4_LAMBDAS)
COV = jnp.diag(LAM)


def sample_population(key, n):
    z = jax.random.normal(key, (n, 2))
    return MU + z * jnp.sqrt(LAM)


def main():
    sched = make_vp_schedule(T=T)
    key = jax.random.PRNGKey(0)
    key, k_model, k_data = jax.random.split(key, 3)
    params = init_model(k_model, in_dim=2, hidden=64, time_dim=32, n_hidden=2)
    opt = adam_init(params)

    step = jax.jit(lambda p, o, x, k: train_step(p, o, x, k, sched, LR))
    for i in range(STEPS):
        k_data, k_batch = jax.random.split(k_data)
        batch = sample_population(k_batch, BATCH)
        params, opt, loss, key = step(params, opt, batch, key)
        if i % 500 == 0 or i == STEPS - 1:
            print(f"step {i:4d}  eps-loss {float(loss):.4f}")

    key, k_xt = jax.random.split(key)
    x_T = jax.random.normal(k_xt, (N_SEEDS, 2))
    learned = make_learned_x0_fn(params, sched)
    exact = make_gaussian_x0_fn(MU, COV, sched)

    pm_learned = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, learned, sched)))(x_T)
    pm_exact = jax.jit(jax.vmap(lambda x: sample_posterior_mean(x, exact, sched)))(x_T)
    ddim_l = jax.jit(jax.vmap(lambda x: sample_ddim(x, learned, sched)))(x_T)
    key, k_st = jax.random.split(key)
    keys = jax.random.split(k_st, N_SEEDS)
    stoch_l = jax.jit(jax.vmap(lambda x, k: sample_stochastic(x, learned, sched, k)))(
        x_T, keys
    )

    def report(name, xs):
        spread, offset = spread_offset(xs, MU)
        print(f"{name:28s}  spread {float(spread):.4f}  offset {float(offset):.4f}")
        return float(spread), float(offset)

    print("--- endpoints vs population mean ---")
    s_pm, _ = report("posterior-mean (learned)", pm_learned)
    report("posterior-mean (exact)", pm_exact)
    s_ddim, _ = report("DDIM eta=0 (learned)", ddim_l)
    s_stoch, _ = report("stochastic DDPM (learned)", stoch_l)
    if s_pm < s_stoch and s_pm < s_ddim:
        print("learned posterior-mean collapsed relative to DDIM and stochastic DDPM.")
    else:
        print("learned posterior-mean did not collapse more than the other samplers.")
    print("done.")


if __name__ == "__main__":
    main()
