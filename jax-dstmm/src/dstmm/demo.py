"""Fit a DStMM to data drawn from a known DStMM and print what came out.

Run with: python -m dstmm.demo
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from dstmm.mcem import mcem_fit
from dstmm.model import (
    bic,
    init_from_data,
    layer_responsibilities,
    log_likelihood,
    n_params,
    sample_dstmm,
)
from dstmm.simulate import paper_style_truth


def adjusted_rand_index(a, b):
    """Plain ARI, so the demo does not need sklearn."""
    a, b = np.asarray(a), np.asarray(b)
    table = np.zeros((a.max() + 1, b.max() + 1))
    for i, j in zip(a, b):
        table[i, j] += 1
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(table.sum())
    expected = sum_i * sum_j / total
    return (sum_ij - expected) / (0.5 * (sum_i + sum_j) - expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--kappa", type=float, default=2.0, help="skewness magnitude of Section 4.1")
    ap.add_argument("--nu", type=float, default=6.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warm", type=int, default=20, help="symmetric deep-t iterations first")
    ap.add_argument("--mc", type=int, default=2, help="Monte Carlo size M of Section 3")
    ap.add_argument("--starts", type=int, default=4, help="random starts, best log-likelihood wins")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    truth = paper_style_truth(kappa=args.kappa, nu=args.nu)
    keys = jax.random.split(jax.random.key(args.seed), 2 + args.starts)
    y, path_index, _, _ = sample_dstmm(keys[0], truth, args.n)
    layer1_true = np.asarray(path_index) // truth.ks[1]

    print(f"p={truth.dims[0]} r={truth.dims[1:]} K={truth.ks} n={args.n} "
          f"nu={args.nu} kappa={args.kappa}")
    print(f"log-likelihood at the truth: {float(log_likelihood(truth, y)):.1f}")

    # Section 3.6: the deep-mixture likelihood is multimodal, so run several
    # starts, warm-start each skewed fit from its own symmetric fit, and keep the
    # one with the largest observed-data log-likelihood.
    t0 = time.time()
    fit = None
    for start in range(args.starts):
        k = jax.random.split(keys[2 + start], 3)
        init = init_from_data(k[0], y, truth.dims, truth.ks, active_set=(), nu0=10.0)
        warm = mcem_fit(k[1], init, y, n_iter=args.warm, active_set=(), n_mc=args.mc)
        cand = mcem_fit(k[2], warm.model, y, n_iter=args.iters, active_set=(1,), n_mc=args.mc)
        print(f"  start {start}: log-likelihood {float(cand.log_lik[-1]):.1f}")
        if fit is None or float(cand.log_lik[-1]) > float(fit.log_lik[-1]):
            fit = cand
    elapsed = time.time() - t0

    trace = np.asarray(fit.log_lik)
    print(f"MCEM: {args.starts} starts of {args.warm} symmetric + {args.iters} skewed "
          f"iterations in {elapsed:.1f}s")
    print(f"log-likelihood {trace[0]:.1f} -> {trace[-1]:.1f}")
    print(f"nu: true {args.nu}, fitted {float(fit.model.nu[0]):.3f}")

    d_hat = np.asarray(fit.model.delta[0])
    d_true = np.asarray(truth.delta[0])
    rmse = np.sqrt(((d_hat - d_true) ** 2).sum() / (2 * truth.dims[0]))
    print(f"first-layer skewness RMSE: {rmse:.4f} (null value {args.kappa / np.sqrt(truth.dims[0]):.4f})")

    layer1_hat = np.asarray(jnp.argmax(layer_responsibilities(fit.model, fit.responsibilities, 1), axis=1))
    print(f"first-layer ARI: {adjusted_rand_index(layer1_true, layer1_hat):.3f}")

    d = n_params(truth.dims, truth.ks, active_set=(1,), df_mode="common")
    print(f"d_code = {d}, BIC = {float(bic(fit.model, y, active_set=(1,))):.1f}")


if __name__ == "__main__":
    main()
