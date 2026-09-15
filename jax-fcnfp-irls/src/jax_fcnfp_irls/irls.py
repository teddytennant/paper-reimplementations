"""Algorithm 1: IRLS with Lasry-Lions continuation (paper Sec. V)."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from jax_fcnfp_irls.ssn import dual_ssn
from jax_fcnfp_irls.surrogate import (
    R_FLOOR,
    admissible_mu,
    irls_weights,
    j_ll,
    j_true,
    log_tail_alpha,
    mu_max,
)

LAM_MIN = 1e-9
STAGE_QP_CAP = 6
FLOW_CHANGE_STAGE = 1e-3
TRUE_TAU = 1e-6
ACCEPT_BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _as_f64(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _np(x):
    return np.asarray(x, dtype=np.float64)


@dataclass
class IRLSResult:
    flow: jnp.ndarray
    j_true: float
    lam: float
    mu: float
    n_iters: int
    incumbent_history: list[float] = field(default_factory=list)
    feasible: bool = True


def flow_gap(f, floor: float = LAM_MIN):
    """Sort |f| descending; K_gap = argmax_j |f|_(j)/|f|_(j+1).

    Returns K_gap (1-based support size), rho_gap, f_up, f_down.
    Magnitudes are floored at `floor` before ratios (paper supplement).
    """
    mag = np.sort(np.maximum(np.abs(_np(f)), floor))[::-1]
    m = mag.size
    if m <= 1:
        return 1, np.inf, float(mag[0]) if m else 0.0, floor
    ratios = mag[:-1] / mag[1:]
    j = int(np.argmax(ratios))
    k_gap = j + 1
    f_up = float(mag[j])
    f_down = float(mag[j + 1])
    rho = f_up / f_down if f_down > 0 else np.inf
    return k_gap, float(rho), f_up, f_down


def _b_rep(f, b, f_up, f_down):
    """Flow-weighted average of b_e near the gap; else median of positive b."""
    f = _np(f)
    b = _np(b)
    ax = np.abs(f)
    near = (ax >= f_down / 2.0) & (ax <= 2.0 * f_up) & (b > 1e-3)
    if np.any(near) and np.sum(ax[near]) > 0:
        return float(np.sum(b[near] * ax[near]) / np.sum(ax[near]))
    pos = b[b > 0]
    if pos.size:
        return float(np.median(pos))
    return 1.0


def accept_step(f, f_trial, r, b, lam, mu, alpha, tol: float = 0.0):
    """Accept f_trial only if J_LL does not increase beyond `tol`; else backtrack.

    Trial points are (1-beta) f + beta f_trial for beta in {1, 1/2, 1/4, ...}.
    If none accepted, keep f.
    """
    f = _as_f64(f)
    f_trial = _as_f64(f_trial)
    j_cur = float(j_ll(f, r, b, lam, mu, alpha))
    slack = tol
    for beta in ACCEPT_BETAS:
        f_new = (1.0 - beta) * f + beta * f_trial
        j_new = float(j_ll(f_new, r, b, lam, mu, alpha))
        if j_new <= j_cur + slack:
            return f_new, True, float(beta)
    return f, False, 0.0


def initialize_smoothing(f, r, b, start: str = "auto"):
    """Sparse or dense lambda0 / mu0 (paper Sec. V and supplement Table IV).

    sparse: lambda0 = 1e-2 * max{f_med^2 / (2 b_med), 1}, theta0 = 0.3
    dense:  lambda0 = 1e2 * max{max |f_e|, 1}, theta0 = 0.7
    mu0 = min{theta0 * lambda0, min_e mu_max,e(lambda0)}
    """
    f = _np(f)
    r = _np(r)
    b = _np(b)
    ax = np.abs(f)
    if start == "auto":
        n_big = int(np.sum(ax > 1.0))
        start = "sparse" if n_big >= 2 else "dense"

    if start == "dense":
        lam0 = 1e2 * max(float(np.max(ax)), 1.0)
        theta0 = 0.7
    else:
        big = ax[ax > 1.0]
        if big.size >= 1:
            f_med = float(np.median(big))
            pos_b = b[b > 0]
            b_med = float(np.median(pos_b)) if pos_b.size else 1.0
            b_med = max(b_med, 1e-12)
            lam0 = 1e-2 * max(f_med * f_med / (2.0 * b_med), 1.0)
        else:
            j0 = float(j_true(f, r, b, tau=TRUE_TAU))
            lam0 = 1e-2 * max(3.0 * j0, 1.0)
        theta0 = 0.3

    alpha0 = float(log_tail_alpha(lam0, lam0))
    mu0 = admissible_mu(r, b, lam0, alpha0, theta=theta0)
    return float(lam0), float(mu0), start


def continuation_update(
    lam,
    f,
    f_prev,
    r,
    b,
    lam0,
    qp_in_stage: int,
    rejected: bool,
    k_gap: int,
    rho_gap: float,
    f_up: float,
    f_down: float,
):
    """Within-stage and across-stage lambda updates (paper Sec. V).

    within: lambda+ = max{min{lambda, lambda_gap}, lambda/5}
    across: lambda+ = max{min{lambda/4, lambda_gap}, lambda/100, 1e-9}
    New stage if relative flow change < 1e-3, line search rejects, or 6 QP solves.
    Then mu = min{lambda/2, min mu_max}.
    """
    lam = float(lam)
    f = _as_f64(f)
    f_prev = _as_f64(f_prev)
    denom = max(float(jnp.linalg.norm(f_prev)), 1e-12)
    rel = float(jnp.linalg.norm(f - f_prev)) / denom
    new_stage = rel < FLOW_CHANGE_STAGE or rejected or (qp_in_stage >= STAGE_QP_CAP)

    b_rep = _b_rep(f, b, f_up, f_down)
    lam_gap = (f_down * f_down) / (2.0 * max(b_rep, 1e-12))

    if new_stage:
        lam_new = max(min(lam / 4.0, lam_gap), lam / 100.0, LAM_MIN)
        qp_next = 1
    else:
        lam_new = max(min(lam, lam_gap), lam / 5.0)
        qp_next = qp_in_stage + 1
    lam_new = max(float(lam_new), LAM_MIN)

    alpha = float(log_tail_alpha(lam_new, lam0))
    r_use = jnp.maximum(_as_f64(r), R_FLOOR)
    mu_new = admissible_mu(r_use, b, lam_new, alpha, theta=0.5)
    return lam_new, mu_new, new_stage, qp_next, rel


def refit_support(C, d, c, r, b, f, k_gap, rho_gap, f_up, f_down, n_nodes: int):
    """Optional linear-cost refit on S = {e : |f_e| > sqrt(f_up f_down)}.

    Triggered when rho_gap >= 10 and 0 < K_gap <= max(4n, 1024).
    Solved as a weighted QP with w_e ≈ r_e / max(|f_e|, eps) on S (and huge
    weights off S), reusing DualSSN. Skipped when the restricted support is empty.
    """
    m = int(_np(f).size)
    if not (rho_gap >= 10.0 and 0 < k_gap <= max(4 * n_nodes, 1024)):
        return None
    thresh = np.sqrt(max(f_up * f_down, 0.0))
    ax = np.abs(_np(f))
    on = ax > thresh
    if not np.any(on):
        return None
    r = _np(r)
    w = np.full(m, 1e12, dtype=np.float64)
    w[on] = np.maximum(r[on], R_FLOOR) / np.maximum(ax[on], 1e-8)
    f_ref, _y, res, _k = dual_ssn(C, d, c, jnp.asarray(w), f_current=f)
    if res > 1e-5:
        return None
    return jnp.clip(f_ref, 0.0, _as_f64(c))


def irls(
    C,
    d,
    c,
    r,
    b,
    f0=None,
    start: str = "auto",
    max_iters: int = 200,
    lam_min: float = LAM_MIN,
    do_refit: bool = True,
    accept_tol: float = 0.0,
):
    """Algorithm 1 IRLS with smoothing continuation.

    InitializeSmoothing; loop Weights, DualSSN, Accept, FlowGap,
    Best(+optional Refit), Continuation. Stop if rho_gap >= 100 and K_gap
    stable 4 iters and incumbent unimproved 8 iters, or lambda <= lam_min.
    """
    C = _as_f64(C)
    d = _as_f64(d)
    c = _as_f64(c)
    r = _as_f64(r)
    b = _as_f64(b)
    n_nodes = int(C.shape[0])

    if f0 is None:
        f, _y, _res, _k = dual_ssn(C, d, c, jnp.ones(c.shape[0], dtype=jnp.float64))
    else:
        f = jnp.clip(_as_f64(f0), 0.0, c)

    lam0, mu, start_used = initialize_smoothing(f, r, b, start=start)
    lam = lam0
    alpha = float(log_tail_alpha(lam, lam0))

    def _score(flow):
        return float(j_true(flow, r, b, tau=TRUE_TAU))

    def _feasible(flow):
        res = float(jnp.linalg.norm(C @ flow - d, ord=jnp.inf))
        flow_np = _np(flow)
        c_np = _np(c)
        return (
            res <= 1e-5
            and np.all(flow_np >= -1e-6)
            and np.all(flow_np <= c_np + 1e-6)
        )

    f_star = f
    j_star = _score(f) if _feasible(f) else np.inf
    history = [j_star] if np.isfinite(j_star) else []
    unimproved = 0
    k_gap_run = 0
    k_gap_prev = None
    qp_in_stage = 1
    y_warm = None

    for it in range(max_iters):
        r_use = jnp.maximum(r, R_FLOOR)
        alpha = float(log_tail_alpha(lam, lam0))
        w = irls_weights(f, r_use, b, lam, mu, alpha, clip=True)
        f_trial, y_warm, _res, _k = dual_ssn(
            C, d, c, w, y0=y_warm, f_current=f
        )
        f_new, accepted, beta = accept_step(
            f, f_trial, r_use, b, lam, mu, alpha, tol=accept_tol
        )
        if beta < 1.0:
            y_warm = None

        k_gap, rho_gap, f_up, f_down = flow_gap(f_new)
        if k_gap_prev is not None and k_gap == k_gap_prev:
            k_gap_run += 1
        else:
            k_gap_run = 1
        k_gap_prev = k_gap

        candidates = [f_new]
        if do_refit:
            f_ref = refit_support(
                C, d, c, r, b, f_new, k_gap, rho_gap, f_up, f_down, n_nodes
            )
            if f_ref is not None:
                candidates.append(f_ref)

        improved = False
        for cand in candidates:
            if not _feasible(cand):
                continue
            jc = _score(cand)
            if jc < j_star - max(1e-12, 1e-9 * abs(j_star) if np.isfinite(j_star) else 0.0):
                f_star = cand
                j_star = jc
                improved = True
        if improved:
            history.append(j_star)
            unimproved = 0
        else:
            unimproved += 1
            if history:
                history.append(history[-1])

        if rho_gap >= 100.0 and k_gap_run >= 4 and unimproved >= 8:
            f = f_new
            break
        if lam <= lam_min:
            f = f_new
            break

        lam, mu, _new_stage, qp_in_stage, _rel = continuation_update(
            lam,
            f_new,
            f,
            r,
            b,
            lam0,
            qp_in_stage,
            rejected=not accepted,
            k_gap=k_gap,
            rho_gap=rho_gap,
            f_up=f_up,
            f_down=f_down,
        )
        f = f_new
        _ = start_used
        _ = it

    if _feasible(f_star) and np.isfinite(j_star):
        flow_out = jnp.clip(f_star, 0.0, c)
        j_out = j_star
        feasible = True
    else:
        flow_out = jnp.clip(f, 0.0, c)
        j_out = _score(flow_out)
        feasible = _feasible(flow_out)

    return IRLSResult(
        flow=flow_out,
        j_true=float(j_out),
        lam=float(lam),
        mu=float(mu),
        n_iters=it + 1,
        incumbent_history=history,
        feasible=feasible,
    )
