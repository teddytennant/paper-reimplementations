"""MSVRM-v1/v2/v3 step mechanics and a full-batch descent check."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_msvr.estimator import correction_gamma
from jax_msvr.metrics import objective
from jax_msvr.msvrm import (
    init_msvrm_state,
    msvrm_v1_step,
    msvrm_v2_step,
    msvrm_v3_step,
)
from jax_msvr.problems import make_quadratic


def _problem_and_state(m=4, d=6, k=2, sigma=0.0, n=None, seed=0):
    problem, _w_star = make_quadratic(
        m, d, k, jax.random.PRNGKey(seed), sigma=sigma, n=n, scale=0.4
    )
    w = jnp.ones(d)
    state = init_msvrm_state(w, problem=problem)
    return problem, state


def test_v1_step_changes_w_by_minus_eta_z():
    problem, state = _problem_and_state()
    eta = 0.05
    new_state = msvrm_v1_step(
        state,
        problem,
        jax.random.PRNGKey(1),
        B1=2,
        alpha=0.3,
        beta=0.3,
        gamma=1.0,
        eta=eta,
        adaptive=False,
    )
    np.testing.assert_allclose(
        np.asarray(new_state.w),
        np.asarray(state.w) - eta * np.asarray(new_state.z),
        rtol=1e-10,
        atol=1e-10,
    )


def test_v1_adaptive_step_uses_normalized_z():
    problem, state = _problem_and_state()
    eta, eps = 0.05, 1e-8
    new_state = msvrm_v1_step(
        state,
        problem,
        jax.random.PRNGKey(2),
        B1=2,
        alpha=0.3,
        beta=0.3,
        gamma=1.0,
        eta=eta,
        adaptive=True,
        eps=eps,
    )
    z = np.asarray(new_state.z)
    expected = np.asarray(state.w) - eta * z / (np.linalg.norm(z) + eps)
    np.testing.assert_allclose(np.asarray(new_state.w), expected, rtol=1e-10, atol=1e-10)


def test_v2_step_changes_w_by_minus_eta_z():
    problem, state = _problem_and_state()
    eta = 0.02
    new_state = msvrm_v2_step(
        state,
        problem,
        jax.random.PRNGKey(3),
        B1=2,
        alpha=0.2,
        beta=0.3,
        gamma=1.0,
        eta=eta,
    )
    np.testing.assert_allclose(
        np.asarray(new_state.w),
        np.asarray(state.w) - eta * np.asarray(new_state.z),
        rtol=1e-10,
        atol=1e-10,
    )


def test_v3_step_changes_w_by_minus_eta_z():
    problem, state = _problem_and_state(n=5)
    eta = 0.02
    new_state = msvrm_v3_step(
        state,
        problem,
        jax.random.PRNGKey(4),
        B1=2,
        alpha=0.2,
        beta=0.3,
        gamma=1.0,
        eta=eta,
        I=2,
    )
    np.testing.assert_allclose(
        np.asarray(new_state.w),
        np.asarray(state.w) - eta * np.asarray(new_state.z),
        rtol=1e-10,
        atol=1e-10,
    )


def test_v2_v3_adaptive_flag():
    problem, state = _problem_and_state(n=4)
    eta, eps = 0.1, 1e-8
    s2 = msvrm_v2_step(
        state,
        problem,
        jax.random.PRNGKey(5),
        B1=2,
        alpha=0.4,
        beta=0.3,
        gamma=0.5,
        eta=eta,
        adaptive=True,
        eps=eps,
    )
    z2 = np.asarray(s2.z)
    np.testing.assert_allclose(
        np.asarray(s2.w),
        np.asarray(state.w) - eta * z2 / (np.linalg.norm(z2) + eps),
        rtol=1e-10,
        atol=1e-10,
    )
    s3 = msvrm_v3_step(
        state,
        problem,
        jax.random.PRNGKey(6),
        B1=2,
        alpha=0.4,
        beta=0.3,
        gamma=0.5,
        eta=eta,
        I=3,
        adaptive=True,
        eps=eps,
    )
    z3 = np.asarray(s3.z)
    np.testing.assert_allclose(
        np.asarray(s3.w),
        np.asarray(state.w) - eta * z3 / (np.linalg.norm(z3) + eps),
        rtol=1e-10,
        atol=1e-10,
    )


def test_fullbatch_v1_decreases_F():
    m, d, k = 6, 8, 2
    problem, w_star = make_quadratic(
        m, d, k, jax.random.PRNGKey(9), sigma=0.0, n=None, scale=0.25
    )
    w = jnp.zeros(d)
    state = init_msvrm_state(w, problem=problem)
    F0 = float(objective(problem, state.w))
    eta, alpha, beta = 0.2, 0.5, 0.5
    gamma = 0.0
    key = jax.random.PRNGKey(10)
    for _ in range(250):
        key, sub = jax.random.split(key)
        state = msvrm_v1_step(
            state,
            problem,
            sub,
            B1=m,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eta=eta,
        )
    FT = float(objective(problem, state.w))
    assert np.isfinite(F0) and np.isfinite(FT)
    assert FT < F0


def test_shapes_after_three_steps():
    m, d, k, B1 = 5, 7, 3, 2
    problem, _ = make_quadratic(m, d, k, jax.random.PRNGKey(11), sigma=0.05, n=4)
    w = jnp.zeros(d)
    gamma = float(correction_gamma(m, B1, 0.3))
    key = jax.random.PRNGKey(12)
    for step_fn, extra in (
        (msvrm_v1_step, {}),
        (msvrm_v2_step, {}),
        (msvrm_v3_step, {"I": 2}),
    ):
        state = init_msvrm_state(w, problem=problem)
        k_run = key
        for _ in range(3):
            k_run, sub = jax.random.split(k_run)
            state = step_fn(
                state,
                problem,
                sub,
                B1=B1,
                alpha=0.2,
                beta=0.3,
                gamma=gamma,
                eta=0.01,
                **extra,
            )
        assert state.w.shape == (d,)
        assert state.u.shape == (m, k)
        assert state.z.shape == (d,)
        assert state.u_prev.shape == (m, k)
        assert state.w_prev.shape == (d,)
        assert state.g_snap.shape == (m, k)
        assert state.jac_snap.shape == (m, k, d)
