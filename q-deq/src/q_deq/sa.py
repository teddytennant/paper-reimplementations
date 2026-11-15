"""Simulated annealing QUBO solver (Metropolis, geometric cooling).

Operates on binary x in {0,1}^n with energy E(x) = x^T Q x + q^T x.
Does not call Kaiwu CIM.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def qubo_energy(x, Q, q):
    return x @ Q @ x + q @ x


def _one_run(Q, q, key, n_sweeps, t0, t_min):
    n = q.shape[0]
    key, k0 = jax.random.split(key)
    x = jax.random.bernoulli(k0, 0.5, (n,)).astype(q.dtype)
    e = qubo_energy(x, Q, q)
    log_cool = jnp.log(t_min / t0)

    def sweep(i, carry):
        x, key, e = carry
        T = t0 * jnp.exp(log_cool * (i / jnp.maximum(n_sweeps - 1, 1)))
        T = jnp.maximum(T, t_min)
        key, kf, ku = jax.random.split(key, 3)
        j = jax.random.randint(kf, (), 0, n)
        x_new = x.at[j].set(1.0 - x[j])
        e_new = qubo_energy(x_new, Q, q)
        dE = e_new - e
        accept = (dE <= 0.0) | (jax.random.uniform(ku, dtype=q.dtype) < jnp.exp(-dE / T))
        x = jnp.where(accept, x_new, x)
        e = jnp.where(accept, e_new, e)
        return x, key, e

    x, key, e = jax.lax.fori_loop(0, n_sweeps, sweep, (x, key, e))
    return x, e


def sa_solve(
    Q,
    q,
    key,
    n_sweeps=200,
    t0=1.0,
    t_min=1e-3,
    n_restarts=4,
):
    """Return a binary vector minimizing x^T Q x + q^T x (approx)."""
    keys = jax.random.split(key, n_restarts)

    def run(k):
        return _one_run(Q, q, k, n_sweeps, t0, t_min)

    xs, es = jax.vmap(run)(keys)
    i = jnp.argmin(es)
    return xs[i], es[i]
