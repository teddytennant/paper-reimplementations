"""Algebra checks against paper eqs. 3-9. Fail on a wrong gamma or swapped u_t."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_msvr.estimator import (
    correction_gamma,
    finite_sum_g_hat,
    msvr_sp_update,
    msvr_update,
    sox_update,
)
from jax_msvr.msvrm import init_msvrm_state, msvrm_v1_step, msvrm_v2_step
from jax_msvr.problems import QuadraticFCCO


def test_gamma_formula_lemma2():
    m, B1, beta = 10, 2, 0.4
    expected = (10 - 2) / (2 * 0.6) + 0.6
    got = float(correction_gamma(m, B1, beta))
    assert np.isclose(got, expected)
    assert not np.isclose(got, 1.0 - beta)  # not STORM


def test_msvr_eq4_matches_numpy_on_sampled_block():
    rng = np.random.default_rng(0)
    m, k, B1 = 5, 3, 2
    u = rng.normal(size=(m, k))
    g_now = rng.normal(size=(B1, k))
    g_prev = rng.normal(size=(B1, k))
    sampled = np.array([1, 4])
    beta, gamma = 0.4, 1.7
    u_jax = msvr_update(
        jnp.asarray(u),
        jnp.asarray(g_now),
        jnp.asarray(g_prev),
        jnp.asarray(sampled),
        beta,
        gamma,
    )
    u_np = u.copy()
    u_np[sampled] = (1 - beta) * u[sampled] + beta * g_now + gamma * (g_now - g_prev)
    np.testing.assert_allclose(np.asarray(u_jax), u_np, rtol=1e-12, atol=1e-12)


def test_unsampled_blocks_copied_unchanged():
    rng = np.random.default_rng(1)
    m, k = 6, 2
    u = rng.normal(size=(m, k))
    sampled = np.array([0, 3])
    g_now = rng.normal(size=(2, k))
    g_prev = rng.normal(size=(2, k))
    u_new = np.asarray(
        msvr_update(
            jnp.asarray(u),
            jnp.asarray(g_now),
            jnp.asarray(g_prev),
            jnp.asarray(sampled),
            0.3,
            2.0,
        )
    )
    unsampled = np.setdiff1d(np.arange(m), sampled)
    np.testing.assert_allclose(u_new[unsampled], u[unsampled])
    assert not np.allclose(u_new[sampled], u[sampled])


def test_sox_is_msvr_with_gamma_zero():
    rng = np.random.default_rng(2)
    m, k, B1 = 4, 2, 2
    u = rng.normal(size=(m, k))
    g_now = rng.normal(size=(B1, k))
    g_prev = rng.normal(size=(B1, k))
    sampled = np.array([0, 2])
    beta = 0.25
    sox = sox_update(jnp.asarray(u), jnp.asarray(g_now), jnp.asarray(sampled), beta)
    msvr = msvr_update(
        jnp.asarray(u),
        jnp.asarray(g_now),
        jnp.asarray(g_prev),
        jnp.asarray(sampled),
        beta,
        0.0,
    )
    np.testing.assert_allclose(np.asarray(sox), np.asarray(msvr), rtol=1e-12, atol=1e-12)


def test_msvr_sp_linearization_matches_numpy():
    rng = np.random.default_rng(3)
    m, k, d, B1 = 4, 2, 5, 2
    u = rng.normal(size=(m, k))
    g_now = rng.normal(size=(B1, k))
    jac = rng.normal(size=(B1, k, d))
    dw = rng.normal(size=(d,))
    sampled = np.array([1, 3])
    beta, gamma = 0.2, 3.1
    u_jax = msvr_sp_update(
        jnp.asarray(u),
        jnp.asarray(g_now),
        jnp.asarray(jac),
        jnp.asarray(dw),
        jnp.asarray(sampled),
        beta,
        gamma,
    )
    linear = np.einsum("bkd,d->bk", jac, dw)
    u_np = u.copy()
    u_np[sampled] = (1 - beta) * u[sampled] + beta * g_now + gamma * linear
    np.testing.assert_allclose(np.asarray(u_jax), u_np, rtol=1e-12, atol=1e-12)


def _fullbatch_problem():
    rng = np.random.default_rng(4)
    m, k, d = 3, 2, 4
    A = rng.normal(size=(m, k, d))
    b = rng.normal(size=(m, k))
    return QuadraticFCCO(A, b, sigma=0.0)


def test_v1_z_uses_grad_f_of_u_prev_not_u_t():
    problem = _fullbatch_problem()
    m, k, d = problem.m, problem.k, problem.d
    w = jnp.ones(d)
    u = jnp.zeros((m, k))
    z = jnp.zeros(d)
    state = init_msvrm_state(w, problem=problem, u=u, z=z)
    # make w_prev different so the two-point term moves u
    state = state._replace(w_prev=w * 0.3)
    alpha, beta, gamma, eta = 0.5, 0.4, 2.0, 0.01
    key = jax.random.PRNGKey(0)
    new_state = msvrm_v1_step(
        state, problem, key, B1=m, alpha=alpha, beta=beta, gamma=gamma, eta=eta
    )
    # reconstruct z with nabla f(u_{t-1}) vs nabla f(u_t)
    idx = jnp.arange(m)
    g_now, jac_now = problem.probe(state.w, idx, jax.random.split(key)[1])
    # probe key in the step is split from `key` as (k_blocks, k_probe);
    # with B1=m, chain uses all blocks so idx permutation does not matter.
    chain_old = np.einsum("mk,mkd->d", np.asarray(problem.grad_f(state.u)), np.asarray(problem.A)) / m
    chain_new = np.einsum("mk,mkd->d", np.asarray(problem.grad_f(new_state.u)), np.asarray(problem.A)) / m
    z_from_old = (1 - alpha) * np.asarray(state.z) + alpha * chain_old
    z_from_new = (1 - alpha) * np.asarray(state.z) + alpha * chain_new
    assert not np.allclose(np.asarray(new_state.u), np.asarray(state.u))
    assert not np.allclose(z_from_old, z_from_new)
    np.testing.assert_allclose(np.asarray(new_state.z), z_from_old, rtol=1e-10, atol=1e-10)


def test_v2_storm_increment_when_alpha_zero():
    problem = _fullbatch_problem()
    m, k, d = problem.m, problem.k, problem.d
    w = jnp.arange(d, dtype=jnp.float64) * 0.1 + 0.5
    u = jnp.ones((m, k)) * 0.2
    u_prev = jnp.ones((m, k)) * -0.4
    z = jnp.arange(d, dtype=jnp.float64) * 0.01
    state = init_msvrm_state(w, problem=problem, u=u, z=z)
    state = state._replace(u_prev=u_prev, w_prev=w * 0.7)
    alpha, beta, gamma, eta = 0.0, 0.3, 1.5, 0.0
    key = jax.random.PRNGKey(7)
    new_state = msvrm_v2_step(
        state, problem, key, B1=m, alpha=alpha, beta=beta, gamma=gamma, eta=eta
    )
    # B1=m, sigma=0: same xi at both times, jac = A
    chain_now = np.einsum("mk,mkd->d", np.asarray(problem.grad_f(state.u)), np.asarray(problem.A)) / m
    chain_prev = np.einsum("mk,mkd->d", np.asarray(problem.grad_f(state.u_prev)), np.asarray(problem.A)) / m
    storm_inc = chain_now - chain_prev  # alpha=0
    np.testing.assert_allclose(
        np.asarray(new_state.z) - np.asarray(state.z),
        storm_inc,
        rtol=1e-10,
        atol=1e-10,
    )


def test_v3_ghat_unbiased_monte_carlo():
    rng = np.random.default_rng(8)
    m, k, d, n = 2, 1, 3, 20
    A = rng.normal(size=(m, k, d))
    C = rng.normal(size=(m, n, k))
    C = C - C.mean(axis=1, keepdims=True)
    b = rng.normal(size=(m, k))
    problem = QuadraticFCCO(A, b, sigma=0.0, C=C)
    w = jnp.asarray(rng.normal(size=(d,)))
    w_tau = jnp.asarray(rng.normal(size=(d,)))
    g_exact = np.asarray(problem.g_exact(w))
    g_snap_exact = np.asarray(problem.g_exact(w_tau))
    i = 0
    hats = []
    key = jax.random.PRNGKey(11)
    n_mc = 200
    for _ in range(n_mc):
        key, sub = jax.random.split(key)
        idx = jnp.array([i])
        g_now, _ = problem.probe(w, idx, sub)
        g_snap_stoch, _ = problem.probe(w_tau, idx, sub)
        hat = finite_sum_g_hat(g_now, g_snap_stoch, jnp.asarray(g_snap_exact[i : i + 1]))
        hats.append(np.asarray(hat)[0])
    mean_hat = np.mean(hats, axis=0)
    np.testing.assert_allclose(mean_hat, g_exact[i], atol=0.15)
