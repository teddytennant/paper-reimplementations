"""Train the neural rate law and assimilate f0 by backpropagating through the solver.

This is Case B of section 4.5: phi = [theta, f0, c0], minimising the trajectory
MSE of eq (17) over the three standardised predictors. The ground truth here is
synthetic, generated with M1 at its published coefficients, so the recovery check
at the end compares the learned D1(S) against -2.32(1 - S).

Run it with

    python -m hybrid_pbe.train --steps 500
"""

from __future__ import annotations

import argparse
import functools
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .data import Dataset, make_dataset
from .grid import Grid
from .hybrid import LearnableSeed, NeuralDissolution, Standardiser, standardiser_from
from .kinetics import M1, solubility
from .solver import simulate


class HybridModel(eqx.Module):
    """The free variables phi of eqs (18) and (19), bundled together."""

    rate: NeuralDissolution
    seeds: list
    c0: jnp.ndarray


class Scalers(NamedTuple):
    c: Standardiser
    L1: Standardiser
    L2: Standardiser


def build_scalers(dataset: Dataset) -> Scalers:
    """Eq (16) applied per predictor over the whole dataset, ahead of training."""
    cat = lambda attr: jnp.concatenate([getattr(e, attr) for e in dataset.experiments])
    return Scalers(
        c=standardiser_from(cat("c_obs")),
        L1=standardiser_from(cat("L1_obs")),
        L2=standardiser_from(cat("L2_obs")),
    )


def init_model(key, dataset: Dataset, width: int = 20, depth: int = 3) -> HybridModel:
    exps = dataset.experiments
    S_obs = jnp.concatenate([e.c_obs / solubility(e.T_values) for e in exps])
    T_obs = jnp.concatenate([e.T_values for e in exps])
    # Reference rates: mean change in mean length and width per unit time,
    # as described at the end of section 4.4.
    dt_total = exps[0].t[-1] - exps[0].t[0]
    ref1 = jnp.mean(jnp.abs(jnp.stack([e.L1_obs[-1] - e.L1_obs[0] for e in exps]))) / dt_total
    ref2 = jnp.mean(jnp.abs(jnp.stack([e.L2_obs[-1] - e.L2_obs[0] for e in exps]))) / dt_total
    rate = NeuralDissolution(
        key, s_mean=jnp.mean(S_obs), s_std=jnp.std(S_obs) + 1e-6,
        t_mean=jnp.mean(T_obs), t_std=jnp.std(T_obs) + 1e-6,
        ref_rate=jnp.stack([ref1, ref2]), width=width, depth=depth,
    )
    seeds = [LearnableSeed(e.f0_measured, dataset.grid, dataset.seed_volume) for e in exps]
    c0 = jnp.stack([e.c0_measured for e in exps])
    return HybridModel(rate=rate, seeds=seeds, c0=c0)


def _filter_spec(model: HybridModel, assimilate_ic: bool):
    """Which leaves Adam may move: theta only (Case A), or theta, f0, c0 (Case B)."""
    is_arr = eqx.is_inexact_array
    return HybridModel(
        rate=jax.tree_util.tree_map(is_arr, model.rate),
        seeds=jax.tree_util.tree_map(lambda x: is_arr(x) and assimilate_ic, model.seeds),
        c0=is_arr(model.c0) and assimilate_ic,
    )


def predict(model: HybridModel, dataset: Dataset, n_substeps: int = 6):
    out = []
    for i, e in enumerate(dataset.experiments):
        f0 = model.seeds[i]()
        res = simulate(f0, model.c0[i], e.temperature, e.t, model.rate,
                       dataset.grid, n_substeps=n_substeps)
        out.append(res)
    return out


def loss_fn(model: HybridModel, dataset: Dataset, scalers: Scalers, n_substeps: int = 6):
    """Trajectory MSE of eq (17), equally weighted over the three predictors."""
    total = 0.0
    for i, e in enumerate(dataset.experiments):
        f0 = model.seeds[i]()
        res = simulate(f0, model.c0[i], e.temperature, e.t, model.rate,
                       dataset.grid, n_substeps=n_substeps)
        total = total + jnp.mean((scalers.c(res.c) - scalers.c(e.c_obs)) ** 2)
        total = total + jnp.mean((scalers.L1(res.L1) - scalers.L1(e.L1_obs)) ** 2)
        total = total + jnp.mean((scalers.L2(res.L2) - scalers.L2(e.L2_obs)) ** 2)
    return total / (3.0 * len(dataset.experiments))


@functools.lru_cache(maxsize=8)
def _adam(lr: float):
    """Cached so repeated train() calls hit the same jit cache entry."""
    return optax.adam(lr)


@eqx.filter_jit
def _step(diff, static, opt_state, dataset, scalers, opt, n_substeps):
    def wrapped(d):
        return loss_fn(eqx.combine(d, static), dataset, scalers, n_substeps)

    loss, grads = jax.value_and_grad(wrapped)(diff)
    updates, opt_state = opt.update(grads, opt_state, diff)
    return eqx.apply_updates(diff, updates), opt_state, loss


def train(model: HybridModel, dataset: Dataset, scalers: Scalers, steps: int = 500,
          lr: float = 1e-3, n_substeps: int = 6, assimilate_ic: bool = True,
          log_every: int = 0):
    """Adam on phi, eqs (18) and (19). Case A holds f0 and c0 fixed, Case B does not.

    The default lr of 1e-3 is the paper's; the command line uses 1e-2 because
    this synthetic problem is much smaller than their 4.8e5 parameters.
    """
    spec = _filter_spec(model, assimilate_ic)
    opt = _adam(float(lr))
    diff, static = eqx.partition(model, spec)
    opt_state = opt.init(diff)

    history = []
    for i in range(steps):
        diff, opt_state, loss = _step(diff, static, opt_state, dataset, scalers,
                                      opt, n_substeps)
        history.append(float(loss))
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f"step {i:4d}  loss {float(loss):.6f}")
    return eqx.combine(diff, static), jnp.asarray(history)


def learned_rate_curve(model: HybridModel, S_values, T):
    """D1(S), D2(S) from the trained network at a fixed temperature."""
    fn = lambda s: model.rate.rates(s, jnp.asarray(T, dtype=float))
    out = jax.vmap(fn)(jnp.asarray(S_values, dtype=float))
    return out[:, 0], out[:, 1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n1", type=int, default=64)
    p.add_argument("--n2", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--no-assimilate", action="store_true",
                   help="Case A: hold the measured initial conditions fixed")
    args = p.parse_args(argv)

    grid = Grid(n1=args.n1, n2=args.n2, L1_max=1000.0, L2_max=200.0)
    key = jax.random.PRNGKey(args.seed)
    k_data, k_model = jax.random.split(key)
    truth = M1()
    dataset = make_dataset(k_data, grid, truth, noise=args.noise)
    scalers = build_scalers(dataset)
    model = init_model(k_model, dataset, width=args.width, depth=args.depth)

    assimilate = not args.no_assimilate
    print(f"grid {grid.n1}x{grid.n2}, {len(dataset.experiments)} synthetic experiments, "
          f"assimilate initial conditions: {assimilate}")
    t0 = time.time()
    model, history = train(model, dataset, scalers, steps=args.steps, lr=args.lr,
                           assimilate_ic=assimilate, log_every=max(1, args.steps // 10))
    print(f"trained in {time.time() - t0:.1f}s, loss {history[0]:.6f} -> {history[-1]:.6f}")

    S_seen = jnp.concatenate([e.c_obs / solubility(e.T_values)
                              for e in dataset.experiments])
    lo, hi = jnp.percentile(S_seen, jnp.asarray([10.0, 90.0]))
    S = jnp.linspace(0.975, 0.999, 7)
    d1, d2 = learned_rate_curve(model, S, 28.0)
    t1, t2 = truth.rates(S)
    print(f"\nlearned rate at T = 28 C. The data covers S in "
          f"[{float(S_seen.min()):.4f}, {float(S_seen.max()):.4f}], with the "
          f"10th to 90th percentile in [{float(lo):.4f}, {float(hi):.4f}].")
    print("     S      D1 learned    D1 M1      D2 learned    D2 M1   in band")
    for i in range(S.shape[0]):
        band = "  yes" if lo <= S[i] <= hi else "   no"
        print(f"  {S[i]:.4f}   {d1[i]: .5f}   {t1[i]: .5f}   {d2[i]: .5f}   "
              f"{t2[i]: .5f}  {band}")


if __name__ == "__main__":
    main()
