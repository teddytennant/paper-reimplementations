"""Inference-time QRG kernel locked to arXiv:2609.31222.

Displayed equations (Euclidean M = I unless noted):

  (3)  h = M^{-1} J^T c
  residual lift of a quotient increment:
       Δx = J^T (J J^T)^{+} Δq
  (5)  B = ρ ||v||
  (8)  u^quo = - min(1, B / (||h|| + ε)) h
  residual delivery (already-signed lift, same cap):
       Δx_QRG = min(1, ρ ||Δx_s|| / ||Δx_q||) Δx_q
       x' = x + Δx_s + Δx_QRG
  (14) product-budget section and residual caps; residual then spends
       remaining sampler budget so the pair cannot exceed ρ ||Δx_s||
  (15) h(R x) = R h(x), u(R x) = R u(x) for invariant q and equivariant v
"""

from __future__ import annotations

import jax.numpy as jnp

LOCAL = "local"
PREDNEXT = "prednext"


def _as_matrix(J):
    J = jnp.asarray(J)
    if J.ndim == 1:
        return J[None, :]
    return J


def _as_vec(v):
    return jnp.asarray(v).reshape(-1)


def horizontal_lift(J, delta_q, rcond: float = 1e-6):
    """Min-norm horizontal lift Δx = J^T (J J^T)^{+} Δq.

    Uses a stable pseudoinverse of the Gram matrix when J is rank-deficient
    rather than skipping the lift.
    """
    J = _as_matrix(J)
    delta_q = _as_vec(delta_q)
    gram = J @ J.T
    return J.T @ (jnp.linalg.pinv(gram, rcond=rcond) @ delta_q)


def covector_lift(J, c, M_inv=None):
    """Metric-horizontal lift of a quotient covector, paper eq. (3).

    h = M^{-1} J^T c. Euclidean M = I when M_inv is None.
    """
    J = _as_matrix(J)
    c = _as_vec(c)
    h = J.T @ c
    if M_inv is not None:
        h = jnp.asarray(M_inv) @ h
    return h


def sampler_budget(delta_x_s, rho: float = 1.0):
    """Paper eq. (5): B = ρ ||v|| with Euclidean metric."""
    return rho * jnp.linalg.norm(jnp.asarray(delta_x_s))


def budgeted_delivery(delta_x_q, delta_x_s, rho: float = 1.0, eps: float = 1e-12):
    """Sampler-budgeted residual delivery.

    Δx_QRG = min(1, ρ ||Δx_s|| / (||Δx_q|| + ε)) Δx_q
    """
    delta_x_q = jnp.asarray(delta_x_q)
    nq = jnp.linalg.norm(delta_x_q)
    budget = sampler_budget(delta_x_s, rho=rho)
    scale = jnp.minimum(1.0, budget / (nq + eps))
    return jnp.where(nq <= eps, jnp.zeros_like(delta_x_q), scale * delta_x_q)


def budgeted_descent(h, delta_x_s, rho: float = 1.0, eps: float = 1e-12):
    """Paper eq. (8): u = - min(1, B / (||h|| + ε)) h with B = ρ ||v||."""
    return -budgeted_delivery(h, delta_x_s, rho=rho, eps=eps)


def product_budget_delivery(
    delta_x_sec,
    delta_x_res,
    delta_x_s,
    rho_s: float = 1.0,
    rho_r: float = 1.0,
    rho: float = 1.0,
    eps: float = 1e-12,
):
    """Section/residual product budget, paper eq. (14) plus remaining budget.

    Each branch is capped by its own radius. Residual then uses the remaining
    sampler budget so ||u_sec|| + ||u_res|| ≤ ρ ||Δx_s||.
    """
    delta_x_sec = jnp.asarray(delta_x_sec)
    delta_x_res = jnp.asarray(delta_x_res)
    ns = jnp.linalg.norm(jnp.asarray(delta_x_s))
    total = rho * ns
    u_sec = budgeted_delivery(delta_x_sec, delta_x_s, rho=rho_s, eps=eps)
    remaining = jnp.maximum(0.0, total - jnp.linalg.norm(u_sec))
    branch_r = rho_r * ns
    res_budget = jnp.minimum(branch_r, remaining)
    nres = jnp.linalg.norm(delta_x_res)
    scale_r = jnp.minimum(1.0, res_budget / (nres + eps))
    u_res = jnp.where(
        nres <= eps, jnp.zeros_like(delta_x_res), scale_r * delta_x_res
    )
    return u_sec + u_res


def combine_step(x, delta_x_s, delta_x_qrg):
    """x' = x + Δx_s + Δx_QRG."""
    return jnp.asarray(x) + jnp.asarray(delta_x_s) + jnp.asarray(delta_x_qrg)


def evaluation_state(x, delta_x_s, mode: str = LOCAL):
    """Local-QRG uses x. PredNext-QRG uses the frozen sampler's predicted next state."""
    x = jnp.asarray(x)
    if mode == PREDNEXT:
        return x + jnp.asarray(delta_x_s)
    if mode == LOCAL:
        return x
    raise ValueError(f"unknown QRG mode {mode!r}")


def qrg_step(
    x,
    delta_x_s,
    quotient,
    target_q,
    rho: float = 1.0,
    mode: str = LOCAL,
    section=None,
    rho_s: float = 1.0,
    rho_r: float = 1.0,
    rcond: float = 1e-6,
    eps: float = 1e-12,
):
    """One frozen-sampler step plus budgeted QRG.

    `quotient(x) -> (q, J)` with J shaped (m, n*3). `section`, if given, is an
    ambient section displacement (n, 3) or a callable of the evaluation state.
    """
    x = jnp.asarray(x)
    delta_x_s = jnp.asarray(delta_x_s)
    x_eval = evaluation_state(x, delta_x_s, mode=mode)
    q, J = quotient(x_eval)
    delta_q = jnp.asarray(target_q) - jnp.asarray(q)
    dx_q = horizontal_lift(J, delta_q, rcond=rcond).reshape(x.shape)
    if section is None:
        dx_qrg = budgeted_delivery(dx_q, delta_x_s, rho=rho, eps=eps)
    else:
        dx_sec = section(x_eval) if callable(section) else jnp.asarray(section)
        dx_qrg = product_budget_delivery(
            dx_sec,
            dx_q,
            delta_x_s,
            rho_s=rho_s,
            rho_r=rho_r,
            rho=rho,
            eps=eps,
        )
    return combine_step(x, delta_x_s, dx_qrg)
