"""Checks for the unconstrained p=2 solver against arXiv:2609.23557."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from optimal_mvi.halpern import inexact_accelerated_halpern, parameter_schedule
from optimal_mvi.tensor import tensor_step


def F_cubic(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    return x + 0.1 * x ** 3


def _problem():
    x0 = 0.5 * jnp.ones(4, dtype=jnp.float64)
    D = float(jnp.linalg.norm(x0))
    return x0, D, 1.0


def test_schedule():
    L2 = 1.0
    D = 1.0
    T = 6
    q, eta, beta, delta = parameter_schedule(p=2, L_p=L2, D=D, T=T)
    assert q == 2.5
    assert abs(eta[0] - 1.0 / (L2 * D)) < 1e-12
    assert abs(beta[1] - 2.5 / 3.5) < 1e-12
    assert abs(eta[1] / eta[0] - (1.0 + 4.0) / (1.0 + 2.5)) < 1e-12
    assert abs(delta - D / T ** 1.5) < 1e-12


def test_tensor_step_solves_model():
    x0, _, _ = _problem()
    M = 1.0
    y = tensor_step(F_cubic, x0, M)
    s = y - x0
    g = F_cubic(x0)
    H = jax.jacfwd(F_cubic)(x0)
    resid = g + H @ s + (M / 2.0) * jnp.linalg.norm(s) * s
    assert float(jnp.linalg.norm(resid)) < 1e-7


def test_lemma_2_1():
    x0, _, _ = _problem()
    M = 1.0
    L2 = 1.0
    y = tensor_step(F_cubic, x0, M)
    s_norm = float(jnp.linalg.norm(y - x0))
    lhs = float(jnp.linalg.norm(F_cubic(y)))
    assert lhs <= (L2 + M / 2.0) * s_norm ** 2 + 1e-6


def test_tensor_step_nonsymmetric_jacobian():
    # Monotone (PSD symmetric part) but not symmetric: must use a linear solve.
    H = jnp.array([[1.0, 3.0], [-1.0, 2.0]], dtype=jnp.float64)
    g = jnp.array([0.4, -0.2], dtype=jnp.float64)
    x = jnp.zeros(2, dtype=jnp.float64)
    y = tensor_step(None, x, M=1.0, g=g, H=H)
    s = y - x
    resid = g + H @ s + 0.5 * jnp.linalg.norm(s) * s
    assert float(jnp.linalg.norm(resid)) < 1e-7


def test_integration_and_acceptance():
    """Algorithm 1. T=6 and T=8 do not reach 1e-2 (exact resolvents give
    about 0.091 and 0.053). T=20 is the horizon at which ||v|| < 1e-2 holds
    with margin; the threshold itself is not weakened.
    """
    x0, D, L2 = _problem()
    M = L2
    f0 = float(jnp.linalg.norm(F_cubic(x0)))

    short = inexact_accelerated_halpern(F_cubic, x0, T=6, L_p=L2, D=D, M=M, p=2)
    assert short.accepted
    assert all(short.inner_accepted)
    assert np.all(np.isfinite(np.asarray(short.residuals)))
    assert float(short.residuals[-1]) < f0

    mid = inexact_accelerated_halpern(F_cubic, x0, T=8, L_p=L2, D=D, M=M, p=2)
    assert mid.accepted
    assert float(mid.residuals[-1]) < f0

    long = inexact_accelerated_halpern(F_cubic, x0, T=20, L_p=L2, D=D, M=M, p=2)
    assert long.accepted
    assert all(long.inner_accepted)
    assert np.all(np.isfinite(np.asarray(long.residuals)))
    final = float(long.residuals[-1])
    assert final < 1e-2
    assert final < f0
