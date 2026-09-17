"""Array rank / size checks for the MSVR pytree."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_msvr.msvrm import init_msvrm_state
from jax_msvr.problems import make_quadratic


def test_state_shapes_w_u_z():
    m, d, k = 6, 11, 3
    problem, _ = make_quadratic(m, d, k, jax.random.PRNGKey(0))
    w = jnp.zeros(d)
    state = init_msvrm_state(w, problem=problem)
    assert state.w.shape == (d,)
    assert state.u.shape == (m, k)
    assert state.z.shape == (d,)
    assert state.u.ndim == 2
    assert state.w.ndim == 1
    assert state.z.ndim == 1


def test_init_without_problem():
    m, d, k = 4, 5, 2
    w = jnp.ones(d)
    state = init_msvrm_state(w, m=m, k=k)
    assert state.w.shape == (d,)
    assert state.u.shape == (m, k)
    assert state.z.shape == (d,)
    assert state.g_snap.shape == (m, k)
    assert state.jac_snap.shape == (m, k, d)
