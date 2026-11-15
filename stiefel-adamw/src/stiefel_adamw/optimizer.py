"""Stiefel-AdamW on a factorization W = B A (arXiv:2609.21039, Algorithm 1).

B ∈ R^{m×r} is unconstrained (Euclidean AdamW).
A ∈ R^{r×n} is constrained to A A^T = I_r, updated by projecting the
Adam direction onto T_X St(n, r) with X = A^T and retracting (Cayley or QR).

Algorithm 1 omits bias correction and weight decay in the exposition;
both are implemented as in AdamW. Bias correction is on by default so
the method is a drop-in for AdamW. Weight decay on A is included in the
ambient direction before the Stiefel projection; that term is normal to
St(n, r) at X (P_X(X) = 0), so it does not move A along the manifold.

Gradient API
------------
(a) ``step(state, g_b, g_a, ...)`` — user supplies Euclidean factor
    gradients G^B = ∇_B L(BA), G^A = ∇_A L(BA) (paper §3.2).
(b) ``step_from_dW(state, dW, ...)`` — user supplies dL/dW. If L depends
    on W = BA then ∇_B L = (dL/dW) A^T and ∇_A L = B^T (dL/dW).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from stiefel_adamw.geometry import cayley_retract, project, qr_retract

Array = jax.Array


class StiefelAdamWState(NamedTuple):
    """Adam moments and factors for one (B, A) block."""

    B: Array
    A: Array
    m_b: Array
    v_b: Array
    m_a: Array
    v_a: Array
    t: Array


def factor_gradients(dW: Array, B: Array, A: Array) -> tuple[Array, Array]:
    """Convert dL/dW to factor gradients G^B, G^A.

    G^B = (dL/dW) A^T,  G^A = B^T (dL/dW).
    """
    g_b = dW @ A.T
    g_a = B.T @ dW
    return g_b, g_a


def init(B: Array, A: Array, *, project_a: bool = True) -> StiefelAdamWState:
    """Initialize moments at 0. Optionally QR-project A onto A A^T = I_r."""
    if project_a:
        a = qr_retract(A.T, jnp.zeros_like(A.T)).T
    else:
        a = A
    zeros_b = jnp.zeros_like(B)
    zeros_a = jnp.zeros_like(a)
    return StiefelAdamWState(
        B=B,
        A=a,
        m_b=zeros_b,
        v_b=zeros_b,
        m_a=zeros_a,
        v_a=zeros_a,
        t=jnp.array(0, dtype=jnp.int32),
    )


def _retract(x: Array, xi: Array, retraction: str, n_cayley_iters: int) -> Array:
    if retraction == "cayley":
        return cayley_retract(x, xi, alpha=1.0, n_iters=n_cayley_iters)
    if retraction == "qr":
        return qr_retract(x, xi)
    raise ValueError(f"unknown retraction {retraction!r}; use 'cayley' or 'qr'")


@partial(
    jax.jit,
    static_argnames=("bias_correction", "retraction", "n_cayley_iters"),
)
def step(
    state: StiefelAdamWState,
    g_b: Array,
    g_a: Array,
    *,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    bias_correction: bool = True,
    retraction: str = "cayley",
    n_cayley_iters: int = 8,
) -> StiefelAdamWState:
    """One Stiefel-AdamW iteration on a (B, A) pair (Algorithm 1).

    Moments are taken in ambient Euclidean space for both factors.
    B is updated with Euclidean AdamW. For A, the preconditioned direction
    is transposed to X = A^T, projected onto T_X St, and retracted.
    """
    t = state.t + 1
    m_b = beta1 * state.m_b + (1.0 - beta1) * g_b
    v_b = beta2 * state.v_b + (1.0 - beta2) * (g_b * g_b)
    m_a = beta1 * state.m_a + (1.0 - beta1) * g_a
    v_a = beta2 * state.v_a + (1.0 - beta2) * (g_a * g_a)

    t_f = t.astype(m_b.dtype)
    if bias_correction:
        bc1 = 1.0 - beta1**t_f
        bc2 = 1.0 - beta2**t_f
        mhat_b, vhat_b = m_b / bc1, v_b / bc2
        mhat_a, vhat_a = m_a / bc1, v_a / bc2
    else:
        mhat_b, vhat_b = m_b, v_b
        mhat_a, vhat_a = m_a, v_a

    upd_b = mhat_b / (jnp.sqrt(vhat_b) + eps)
    b_new = state.B - lr * (upd_b + weight_decay * state.B)

    # D is r×n like A; geometry is on X = A^T ∈ R^{n×r}.
    upd_a = mhat_a / (jnp.sqrt(vhat_a) + eps)
    x = state.A.T
    d_x = upd_a.T + weight_decay * x
    xi = -lr * project(x, d_x)
    x_new = _retract(x, xi, retraction, n_cayley_iters)
    a_new = x_new.T

    return StiefelAdamWState(
        B=b_new, A=a_new, m_b=m_b, v_b=v_b, m_a=m_a, v_a=v_a, t=t
    )


def step_from_dW(
    state: StiefelAdamWState,
    dW: Array,
    **kwargs,
) -> StiefelAdamWState:
    """Algorithm 1 step from dL/dW instead of factor gradients."""
    g_b, g_a = factor_gradients(dW, state.B, state.A)
    return step(state, g_b, g_a, **kwargs)


def factorization_loss(B: Array, A: Array, W: Array) -> Array:
    """L = (1/2) ||BA - W||_F^2."""
    residual = B @ A - W
    return 0.5 * jnp.sum(residual * residual)


__all__ = [
    "StiefelAdamWState",
    "factor_gradients",
    "factorization_loss",
    "init",
    "step",
    "step_from_dW",
]
