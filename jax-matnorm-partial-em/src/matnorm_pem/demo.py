"""Small end-to-end run: python -m matnorm_pem.demo

Fits the same synthetic data three ways under element-wise missingness, checks
the submatrix specialization against the dense conditional covariance, and
fits a two-component mixture with a quarter of the cells deleted.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from . import opcount, synth
from .density import sample_matnorm
from .em import fit_matnorm, missing_rmse
from .estep import exact_estep
from .mixture import adjusted_rand_index, fit_mixture
from .structural import fit_matnorm_structural, full_V, structural_estep


def _fit(name, Y, mask, Y_complete, **kw):
    opcount.reset()
    t0 = time.perf_counter()
    est, hist = fit_matnorm(Y, mask, **kw)
    dt = time.perf_counter() - t0
    opcount.disable()
    state = exact_estep(Y, mask, est)
    rmse = float(missing_rmse(Y_complete, state.Yhat, mask))
    return {
        "name": name,
        "ll": hist["loglik"][-1],
        "rmse": rmse,
        "estep_factorizations": opcount.total("estep"),
        "max_order": opcount.max_dim("estep"),
        "seconds": dt,
    }


def main():
    p, q, n = 5, 6, 300
    rate = 0.25
    params = synth.paper_params(p, q)
    Y = sample_matnorm(jr.key(0), params, n)
    mask = synth.mcar_mask(jr.key(1), n, p, q, rate)
    print(f"matrix normal, p={p} q={q} N={n}, {int(rate * 100)}% of cells missing at random")
    print(f"true sigma^2 = {float(params.sigma2):.4f}\n")

    rows = [
        _fit("exact EM", Y, mask, Y, method="exact", n_iter=50),
        _fit("partial EM", Y, mask, Y, method="partial", n_iter=50),
        _fit("plug-in", Y, mask, Y, method="plugin", n_iter=50),
    ]
    base = rows[0]["ll"]
    print(f"{'method':<12}{'loglik':>14}{'gap/N':>12}{'RMSE':>9}{'inversions':>12}{'order':>7}{'sec':>7}")
    for r in rows:
        print(f"{r['name']:<12}{r['ll']:>14.4f}{(base - r['ll']) / n:>12.2e}"
              f"{r['rmse']:>9.4f}{r['estep_factorizations']:>12d}{r['max_order']:>7d}"
              f"{r['seconds']:>7.2f}")

    print("\nsubmatrix missingness, 2 x 3 block")
    arow, acol = synth.rect_indicators(jr.key(2), n, p, q, 2, 3)
    bmask = jnp.einsum("np,nq->npq", arow, acol) > 0
    dense = exact_estep(Y, bmask, params)
    _, VR, VC = structural_estep(Y, arow, acol, params, mode="block", mean_sweeps=0)
    V = jax.vmap(full_V, in_axes=(0, 0, None))(VR, VC, params.sigma2)
    print(f"  separable vs dense conditional covariance, max abs difference: "
          f"{float(jnp.max(jnp.abs(V - dense.V))):.2e}")
    opcount.reset()
    _, hist_block = fit_matnorm_structural(Y, arow, acol, n_iter=40, mode="block")
    block_work = opcount.work("estep")
    opcount.reset()
    _, hist_dense = fit_matnorm(Y, bmask, method="exact", n_iter=40)
    dense_work = opcount.work("estep")
    opcount.disable()
    print(f"  loglik  block {hist_block['loglik'][-1]:.4f}   dense {hist_dense['loglik'][-1]:.4f}")
    print(f"  sum of cubed factorization orders  block {block_work}   dense {dense_work}")

    print("\ntwo-component mixture, 25% missing, N=300")
    truth = synth.mixture_params(4, 5, 2)
    Ym, labels = synth.sample_mixture(jr.key(3), truth, 300)
    mmask = synth.mcar_mask(jr.key(4), 300, 4, 5, 0.25)
    for method in ("exact", "partial"):
        t0 = time.perf_counter()
        est, resp, hist = fit_mixture(Ym, mmask, 2, jr.key(5), method=method, n_iter=30)
        dt = time.perf_counter() - t0
        ari = adjusted_rand_index(np.asarray(labels), np.asarray(jnp.argmax(resp, axis=1)))
        print(f"  {method:<8} loglik {hist['loglik'][-1]:>12.4f}   ARI {ari:.3f}   {dt:.1f}s")


if __name__ == "__main__":
    main()
