"""Q-DEQ and Anderson reduce residual on a linear contraction."""

import jax
import jax.numpy as jnp

from q_deq.deq import residual
from q_deq.solver import anderson_solve, qdeq_solve


def _linear_contraction(d, din, rho=0.45):
    A = rho * jnp.eye(d)
    B = 0.25 * jnp.eye(d, din)

    def F(z, x):
        return A @ z + B @ x

    return F, A, B


def test_qdeq_update_reduces_residual_vs_doing_nothing():
    d, din = 8, 6
    F, _A, _B = _linear_contraction(d, din)
    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)
    z0 = jax.random.normal(k1, (d,))
    x = jax.random.normal(k2, (din,))
    r0 = residual(F, z0, x)
    n0 = jnp.linalg.norm(r0)

    z_star = qdeq_solve(
        F, z0, x, k3, k_inner=6, n_dirs=3, n_sweeps=60, n_restarts=2
    )
    r1 = residual(F, z_star, x)
    n1 = jnp.linalg.norm(r1)
    # Doing nothing keeps n0; the Q-DEQ step must improve.
    assert n1 < n0 * 0.95, f"residual {float(n1)} vs start {float(n0)}"
    assert n1 < n0


def test_anderson_mixer_reduces_residual():
    d, din = 8, 6
    F, _A, _B = _linear_contraction(d, din)
    key = jax.random.PRNGKey(11)
    k1, k2 = jax.random.split(key)
    z0 = jax.random.normal(k1, (d,))
    x = jax.random.normal(k2, (din,))
    r0 = residual(F, z0, x)
    n0 = jnp.linalg.norm(r0)

    z_star = anderson_solve(F, z0, x, n_iter=10, m=3, beta=0.8)
    r1 = residual(F, z_star, x)
    n1 = jnp.linalg.norm(r1)
    assert n1 < n0 * 0.5, f"Anderson residual {float(n1)} vs start {float(n0)}"
