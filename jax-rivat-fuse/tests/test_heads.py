"""Quadratic head reductions and task-coupling scalar."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rivat_fuse.heads import (
    init_quadratic_head,
    quadratic_form,
    quadratic_head,
    task_coupling,
)


D = 7
RANK = 3
N_CLASSES = 2


def test_linear_when_S_zero():
    key = jax.random.PRNGKey(0)
    params = init_quadratic_head(key, D, rank=RANK, n_classes=N_CLASSES)
    params = {
        **params,
        "L": jnp.zeros_like(params["L"]),
        "s_diag": jnp.zeros_like(params["s_diag"]),
    }
    h = jax.random.normal(jax.random.PRNGKey(1), (D,))
    scores = quadratic_head(params, h)
    expected = params["w"] @ h + params["b"]
    assert jnp.allclose(scores, expected, atol=1e-5)


def test_pure_quadratic_when_w_b_zero():
    key = jax.random.PRNGKey(2)
    params = init_quadratic_head(key, D, rank=RANK, n_classes=N_CLASSES)
    params = {
        **params,
        "w": jnp.zeros_like(params["w"]),
        "b": jnp.zeros_like(params["b"]),
    }
    h = jax.random.normal(jax.random.PRNGKey(3), (D,))
    scores = quadratic_head(params, h)
    expected = jax.vmap(lambda sd, L: quadratic_form(h, sd, L))(
        params["s_diag"], params["L"]
    )
    assert jnp.allclose(scores, expected, atol=1e-5)
    # Explicit h^T S h with S = Diag(s) + L L^T for class 0.
    S0 = jnp.diag(params["s_diag"][0]) + params["L"][0] @ params["L"][0].T
    assert jnp.allclose(scores[0], h @ S0 @ h, atol=1e-5)


def test_task_coupling_nonneg_for_psd_laplacian():
    T = 3
    W = jax.random.normal(jax.random.PRNGKey(4), (D, T))
    # Path graph Laplacian on 3 tasks.
    A = jnp.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32
    )
    L_T = jnp.diag(A.sum(axis=1)) - A
    omega = task_coupling(W, L_T)
    assert float(omega) >= -1e-5
