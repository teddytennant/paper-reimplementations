"""Tiny CPU smoke: Algorithm 1 on a synthetic SPD pair, then a 1-D ORF sample."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from jax_dwmf.algorithm import (
    diagonal_truncation_transform,
    materialize_N,
    operator_norm,
    true_f_M,
)
from jax_dwmf.random_features import sample_optimized_features, shift_invariant_kernel


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    n = 8
    Gmat = jax.random.normal(key, (n, n))
    A = 0.5 * (Gmat + Gmat.T)
    evals, vecs = jnp.linalg.eigh(A)
    A = (vecs * jnp.clip(evals, 0.0)) @ vecs.T
    d = jnp.asarray([1.6, 1.1, 0.7, 0.05, 0.04, 0.02, 0.01, 0.0])
    gamma = 0.4
    eps_D = 0.1
    out = diagonal_truncation_transform(
        key,
        d,
        A,
        gamma,
        f=lambda x: 1.0 / x,
        eps_D=eps_D,
        exact_support=True,
    )
    N = materialize_N(out)
    fM = true_f_M(d, A, gamma, lambda x: 1.0 / x)
    err = float(operator_norm(fM - N))
    print(f"Alg. 1: n={n} s={out.s} ||f(M)-N||_op={err:.4e} f(γ)={out.f_gamma:.4f}")

    q = jnp.asarray([0.5, 0.25, 0.15, 0.07, 0.03])
    Q = jnp.asarray([1.0, 0.5, 0.25, 0.1, 0.05])
    k = shift_invariant_kernel(Q)
    samples, w, repr_N = sample_optimized_features(
        key, q, Q, eps_reg=0.3, eps_D=0.2, n_samples=5, k=k
    )
    print(f"ORF: samples={list(map(int, samples))} s={repr_N.s} w={np_round(w)}")


def np_round(w):
    return [float(f"{float(x):.4f}") for x in w]


if __name__ == "__main__":
    main()
