"""Algorithm 1: one-step constraint, gradient conversion, descent, gauge."""

import jax
import jax.numpy as jnp
import pytest

from stiefel_adamw.geometry import random_stiefel, row_orth_error
from stiefel_adamw.optimizer import (
    factor_gradients,
    factorization_loss,
    init,
    step,
    step_from_dW,
)

jax.config.update("jax_enable_x64", True)


def _random_factors(key, m, n, r, dtype=jnp.float64):
    k1, k2 = jax.random.split(key)
    b = jax.random.normal(k1, (m, r), dtype=dtype)
    a = random_stiefel(k2, n, r, dtype=dtype).T
    return b, a


def _naive_adamw_step(B, A, m_b, v_b, m_a, v_a, t, g_b, g_a, lr, beta1, beta2, eps, wd, bias_correction):
    t = t + 1
    m_b = beta1 * m_b + (1.0 - beta1) * g_b
    v_b = beta2 * v_b + (1.0 - beta2) * (g_b * g_b)
    m_a = beta1 * m_a + (1.0 - beta1) * g_a
    v_a = beta2 * v_a + (1.0 - beta2) * (g_a * g_a)
    t_f = t.astype(B.dtype)
    if bias_correction:
        bc1, bc2 = 1.0 - beta1**t_f, 1.0 - beta2**t_f
        mhb, vhb, mha, vha = m_b / bc1, v_b / bc2, m_a / bc1, v_a / bc2
    else:
        mhb, vhb, mha, vha = m_b, v_b, m_a, v_a
    B = B - lr * (mhb / (jnp.sqrt(vhb) + eps) + wd * B)
    A = A - lr * (mha / (jnp.sqrt(vha) + eps) + wd * A)
    return B, A, m_b, v_b, m_a, v_a, t


@pytest.mark.parametrize("retraction", ["qr", "cayley"])
@pytest.mark.parametrize("bias_correction", [True, False])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_one_step_preserves_stiefel(retraction, bias_correction, weight_decay):
    m, n, r = 8, 10, 3
    B, A = _random_factors(jax.random.key(4), m, n, r)
    assert row_orth_error(A) < 1e-8
    state = init(B, A, project_a=False)
    g_b = jax.random.normal(jax.random.key(5), B.shape, dtype=B.dtype)
    g_a = jax.random.normal(jax.random.key(6), A.shape, dtype=A.dtype)
    state = step(
        state,
        g_b,
        g_a,
        lr=1e-2,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
        retraction=retraction,
        n_cayley_iters=10,
    )
    assert row_orth_error(state.A) < 1e-5


def test_gradient_conversion_matches_autodiff():
    m, n, r = 7, 9, 4
    B, A = _random_factors(jax.random.key(7), m, n, r)
    W = jax.random.normal(jax.random.key(8), (m, n), dtype=jnp.float64)

    def loss(b, a):
        return factorization_loss(b, a, W)

    g_b_ad, g_a_ad = jax.grad(loss, argnums=(0, 1))(B, A)
    dW = B @ A - W  # dL/dW for L = 1/2 ||BA - W||^2
    g_b, g_a = factor_gradients(dW, B, A)
    assert jnp.allclose(g_b, g_b_ad, atol=1e-8)
    assert jnp.allclose(g_a, g_a_ad, atol=1e-8)
    assert jnp.allclose(g_b, dW @ A.T, atol=1e-10)
    assert jnp.allclose(g_a, B.T @ dW, atol=1e-10)


@pytest.mark.parametrize("retraction", ["qr", "cayley"])
@pytest.mark.parametrize("bias_correction", [True, False])
def test_descent_on_quadratic(retraction, bias_correction):
    m, n, r = 12, 12, 4
    key = jax.random.key(9)
    k1, k2, k3 = jax.random.split(key, 3)
    B, A = _random_factors(k1, m, n, r)
    W = jax.random.normal(k2, (m, r), dtype=jnp.float64) @ random_stiefel(
        k3, n, r, dtype=jnp.float64
    ).T
    state = init(B, A, project_a=False)
    loss0 = factorization_loss(state.B, state.A, W)

    def body(state, _):
        dW = state.B @ state.A - W
        state = step_from_dW(
            state,
            dW,
            lr=5e-3,
            weight_decay=0.0,
            bias_correction=bias_correction,
            retraction=retraction,
            n_cayley_iters=8,
        )
        return state, None

    state, _ = jax.lax.scan(body, state, None, length=50)
    loss1 = factorization_loss(state.B, state.A, W)
    assert loss1 < loss0
    assert row_orth_error(state.A) < 1e-5


def test_gauge_stiefel_blocks_scaling():
    m, n, r = 16, 16, 4
    key = jax.random.key(10)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B0, A_st = _random_factors(k1, m, n, r)
    W = jax.random.normal(k2, (m, r), dtype=jnp.float64) @ random_stiefel(
        k3, n, r, dtype=jnp.float64
    ).T

    state = init(B0, A_st, project_a=False)

    def body_st(state, _):
        dW = state.B @ state.A - W
        return step_from_dW(state, dW, lr=1e-2, retraction="qr", bias_correction=True), None

    state, _ = jax.lax.scan(body_st, state, None, length=200)

    cond_st = jnp.linalg.cond(state.A)
    assert row_orth_error(state.A) < 1e-5
    assert cond_st < 1.1
    assert jnp.allclose(jnp.linalg.norm(state.A, "fro") ** 2, jnp.array(r, dtype=A_st.dtype), atol=1e-4)

    scales = jnp.logspace(-1.0, 2.0, r)
    q = random_stiefel(k4, n, r, dtype=jnp.float64).T
    A_naive = scales[:, None] * q
    B_naive = B0 / (jnp.mean(scales) + 1e-12)
    cond_naive0 = jnp.linalg.cond(A_naive)
    assert cond_naive0 > 10.0

    m_b = jnp.zeros_like(B_naive)
    v_b = jnp.zeros_like(B_naive)
    m_a = jnp.zeros_like(A_naive)
    v_a = jnp.zeros_like(A_naive)
    t = jnp.array(0, dtype=jnp.int32)

    def body_n(carry, _):
        B, A, m_b, v_b, m_a, v_a, t = carry
        dW = B @ A - W
        g_b, g_a = factor_gradients(dW, B, A)
        out = _naive_adamw_step(
            B, A, m_b, v_b, m_a, v_a, t, g_b, g_a, 1e-2, 0.9, 0.999, 1e-8, 0.0, True
        )
        return out, None

    (B_n, A_n, *_), _ = jax.lax.scan(
        body_n, (B_naive, A_naive, m_b, v_b, m_a, v_a, t), None, length=200
    )
    cond_naive = jnp.linalg.cond(A_n)
    assert cond_naive > 5.0
    assert cond_st < 0.25 * cond_naive
