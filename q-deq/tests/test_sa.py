"""SA finds a known unique minimum of a tiny 4-bit QUBO."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from q_deq.sa import sa_solve, qubo_energy


def _brute_force(Q, q):
    n = int(q.shape[0])
    best_e = np.inf
    best_xs = []
    Qn = np.asarray(Q)
    qn = np.asarray(q)
    for bits in itertools.product([0.0, 1.0], repeat=n):
        x = np.array(bits)
        e = float(x @ Qn @ x + qn @ x)
        if e < best_e - 1e-8:
            best_e = e
            best_xs = [x]
        elif abs(e - best_e) <= 1e-8:
            best_xs.append(x)
    return best_xs, best_e


def test_sa_finds_unique_4bit_minimum():
    Q = jnp.array(
        [
            [0.0, 0.4, -0.2, 0.1],
            [0.4, 0.0, 0.3, -0.5],
            [-0.2, 0.3, 0.0, 0.2],
            [0.1, -0.5, 0.2, 0.0],
        ],
        dtype=jnp.float32,
    )
    q = jnp.array([-1.7, 0.8, -1.4, 1.1], dtype=jnp.float32)
    mins, e_min = _brute_force(Q, q)
    assert len(mins) == 1, f"need unique min, got {len(mins)}"
    target = mins[0]

    key = jax.random.PRNGKey(7)
    x_hat, e_hat = sa_solve(
        Q, q, key, n_sweeps=400, t0=2.0, t_min=1e-4, n_restarts=8
    )
    x_hat = np.asarray(x_hat)
    assert np.allclose(x_hat, target, atol=1e-5), f"got {x_hat}, want {target}"
    assert abs(float(e_hat) - e_min) < 1e-4
    assert abs(float(qubo_energy(jnp.asarray(x_hat), Q, q)) - e_min) < 1e-4
