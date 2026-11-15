"""One fused MoARa training step on a single matrix (Algorithm 1)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from moara.decompose import decompose, repeat_blocks
from moara.project import project, reconstruct_direction, truncated_svd

Array = jnp.ndarray

GALORE_SCALE = 0.25
T_SVD = 200
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-6


class MoARaState(NamedTuple):
    """AdamW states for M and Ṽ, plus the left projector P and step count."""

    step: int
    P: Array
    m_mean: Array
    m_var: Array
    vt_mean: Array
    vt_var: Array


def _adamw(
    grad: Array,
    mean: Array,
    var: Array,
    t: int,
    beta1: float = ADAM_B1,
    beta2: float = ADAM_B2,
    eps: float = ADAM_EPS,
    weight_decay: float = 0.0,
    param: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Bias-corrected AdamW; weight decay is 0 in the paper's low-rank opts."""
    t = int(t)
    mean = beta1 * mean + (1.0 - beta1) * grad
    var = beta2 * var + (1.0 - beta2) * (grad * grad)
    m_hat = mean / (1.0 - beta1**t)
    v_hat = var / (1.0 - beta2**t)
    update = m_hat / (jnp.sqrt(v_hat) + eps)
    if weight_decay != 0.0 and param is not None:
        update = update + weight_decay * param
    return update, mean, var


def init_moara_state(G: Array, block_size: int, rank: int, eps: float = 1e-6, key=None) -> MoARaState:
    """Initialize projector from TruncatedSVD(V, r) and zero Adam moments."""
    _m, v = decompose(G, block_size, eps=eps)
    p = truncated_svd(v, rank, key=key)
    r = int(p.shape[1])
    n = int(v.shape[1])
    k = int(_m.shape[1])
    z_m = jnp.zeros_like(_m)
    z_vt = jnp.zeros((r, n), dtype=v.dtype)
    del k
    return MoARaState(
        step=0,
        P=p,
        m_mean=z_m,
        m_var=jnp.zeros_like(z_m),
        vt_mean=z_vt,
        vt_var=jnp.zeros_like(z_vt),
    )


def moara_update(
    W: Array,
    G: Array,
    state: MoARaState | None = None,
    *,
    block_size: int,
    rank: int,
    lr: float,
    scale: float = GALORE_SCALE,
    t_svd: int = T_SVD,
    eps: float = 1e-6,
    beta1: float = ADAM_B1,
    beta2: float = ADAM_B2,
    adam_eps: float = ADAM_EPS,
    weight_decay: float = 0.0,
    key=None,
) -> tuple[Array, MoARaState]:
    """One Algorithm 1 step for a single target matrix W.

    1. M, V ← Decompose(G, B)
    2. Every T_SVD steps: P ← TruncatedSVD(V, r)
    3. Ṽ ← P^T V
    4. ΔṼ ← AdamW(Ṽ); ΔV ← P ΔṼ; ΔV ← scale · ΔV
    5. ΔM ← AdamW(M)
    6. W ← W − η Rep_B(ΔM) ⊙ ΔV
    """
    W = jnp.asarray(W)
    G = jnp.asarray(G)
    mag, v = decompose(G, block_size, eps=eps)
    n = int(W.shape[1])
    step = 0 if state is None else int(state.step)
    need_svd = state is None or state.P is None or (step % int(t_svd) == 0)
    if need_svd:
        p = truncated_svd(v, rank, key=key)
    else:
        p = state.P
    v_tilde = project(v, p)
    r = int(p.shape[1])
    if state is None or state.vt_mean.shape != v_tilde.shape or state.m_mean.shape != mag.shape:
        m_mean = jnp.zeros_like(mag)
        m_var = jnp.zeros_like(mag)
        vt_mean = jnp.zeros_like(v_tilde)
        vt_var = jnp.zeros_like(v_tilde)
    else:
        m_mean, m_var = state.m_mean, state.m_var
        vt_mean, vt_var = state.vt_mean, state.vt_var

    t = step + 1
    d_vt, vt_mean, vt_var = _adamw(
        v_tilde,
        vt_mean,
        vt_var,
        t,
        beta1=beta1,
        beta2=beta2,
        eps=adam_eps,
        weight_decay=weight_decay,
        param=None,
    )
    d_v = reconstruct_direction(d_vt, p) * scale
    d_m, m_mean, m_var = _adamw(
        mag,
        m_mean,
        m_var,
        t,
        beta1=beta1,
        beta2=beta2,
        eps=adam_eps,
        weight_decay=weight_decay,
        param=None,
    )
    d_w = repeat_blocks(d_m, block_size, n) * d_v
    w_new = W - lr * d_w
    new_state = MoARaState(
        step=t,
        P=p,
        m_mean=m_mean,
        m_var=m_var,
        vt_mean=vt_mean,
        vt_var=vt_var,
    )
    del r
    return w_new, new_state
