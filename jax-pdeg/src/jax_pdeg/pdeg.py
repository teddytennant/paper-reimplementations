"""Projected damped extragradient: Algorithms 1 and 2 of arXiv:2609.17973."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from jax_pdeg.params import PDEGparams, warmup_length
from jax_pdeg.projections import project_unconstrained

Array = jnp.ndarray
GradFn = Callable[[Array, Array], Array]
ProjFn = Callable[[Array], Array]


@dataclass
class PDEGstate:
    x: Array
    y: Array
    z: Array
    xi: Array
    n: Array
    v: Array


@dataclass
class PDEGresult:
    state: PDEGstate
    S: Array
    x_hist: Array
    y_hist: Array
    z_hist: Array
    xi_hist: Array
    n_hist: Array
    v_hist: Array
    x_out: Array
    y_out: Array
    z_out: Array
    gs_index: int
    os_index: int
    n_oracle: int
    T_w: int


def init_state(x0, y0) -> PDEGstate:
    x = jnp.asarray(x0, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y0, dtype=jnp.float64).reshape(-1)
    return PDEGstate(
        x=x,
        y=y,
        z=x + 0,
        xi=jnp.zeros_like(x),
        n=jnp.zeros_like(y),
        v=jnp.zeros_like(y),
    )


def G(f, x, y, z, lam: float, tau: float):
    """Regularized saddle G(x, y; z) from (3.1)."""
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    z = jnp.asarray(z, dtype=jnp.float64).reshape(-1)
    return f(x, y) + 0.5 * lam * jnp.dot(x - z, x - z) - 0.5 * tau * jnp.dot(y, y)


def grad_G(grad_x: GradFn, grad_y: GradFn, x, y, z, lam: float, tau: float):
    """(∇x G, ∇y G) from (3.1)."""
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    z = jnp.asarray(z, dtype=jnp.float64).reshape(-1)
    return grad_x(x, y) + lam * (x - z), grad_y(x, y) - tau * y


def certificate_from_fgrads(gx, gy, y, xi, n, tau: float) -> Array:
    rx = gx + xi
    ry = -gy + tau * y + n
    return jnp.dot(rx, rx) + jnp.dot(ry, ry)


def certificate_S(grad_x: GradFn, grad_y: GradFn, x, y, xi, n, tau: float) -> Array:
    """Game-stationarity certificate S_t from (3.4)."""
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
    xi = jnp.asarray(xi, dtype=jnp.float64).reshape(-1)
    nvec = jnp.asarray(n, dtype=jnp.float64).reshape(-1)
    return certificate_from_fgrads(grad_x(x, y), grad_y(x, y), y, xi, nvec, tau)


def step(
    state: PDEGstate,
    grad_x: GradFn,
    grad_y: GradFn,
    params: PDEGparams,
    project_x: ProjFn,
    project_y: ProjFn,
    beta: Optional[float] = None,
    cached_fgrad: Optional[tuple[Array, Array]] = None,
) -> tuple[PDEGstate, Array, tuple[Array, Array], int]:
    """One iteration of Algorithm 1, lines 3–11.

    If ``cached_fgrad`` is (∇x f, ∇y f) at ``(state.x, state.y)``, that pair is
    not re-queried. Returns ``(new_state, S_t, cache at new pair, n_new_queries)``.
    Each query is one full first-order oracle call (∇x f, ∇y f) at a point.
    """
    lam = params.lam
    tau = params.tau
    h = params.h
    alpha = params.alpha
    omag = (1.0 - params.alpha) * params.gamma
    beta_t = params.beta if beta is None else float(beta)
    n_queries = 0

    if cached_fgrad is None:
        gx = grad_x(state.x, state.y)
        gy = grad_y(state.x, state.y)
        n_queries += 1
    else:
        gx, gy = cached_fgrad

    gxG = gx + lam * (state.x - state.z)
    gyG = gy - tau * state.y
    x_tilde = jnp.asarray(project_x(state.x - h * (gxG + state.xi)), dtype=jnp.float64).reshape(-1)
    y_tilde = jnp.asarray(project_y(state.y + h * (gyG - state.n + state.v)), dtype=jnp.float64).reshape(-1)

    gx_t = grad_x(x_tilde, y_tilde)
    gy_t = grad_y(x_tilde, y_tilde)
    n_queries += 1
    gxG_t = gx_t + lam * (x_tilde - state.z)
    gyG_t = gy_t - tau * y_tilde

    vbar = alpha * state.v + omag * gyG_t
    x_new = jnp.asarray(project_x(state.x - h * gxG_t), dtype=jnp.float64).reshape(-1)
    xi_new = (state.x - x_new) / h - gxG_t
    y_new = jnp.asarray(project_y(state.y + h * (gyG_t + vbar)), dtype=jnp.float64).reshape(-1)
    n_new = ((state.y - y_new) / h + gyG_t + vbar) / (1.0 + omag)

    gx_c = grad_x(x_new, y_new)
    gy_c = grad_y(x_new, y_new)
    n_queries += 1
    gyG_c = gy_c - tau * y_new
    v_new = alpha * state.v + omag * (gyG_c - n_new)
    z_new = state.z + beta_t * (x_new - state.z)

    S = certificate_from_fgrads(gx_c, gy_c, y_new, xi_new, n_new, tau)
    new_state = PDEGstate(x=x_new, y=y_new, z=z_new, xi=xi_new, n=n_new, v=v_new)
    return new_state, S, (gx_c, gy_c), n_queries


def run_pdeg(
    grad_x: GradFn,
    grad_y: GradFn,
    x0,
    y0,
    params: PDEGparams,
    project_x: ProjFn | None = None,
    project_y: ProjFn | None = None,
    T: int = 1,
    T_w: int = 0,
    key=None,
    os_index: int | None = None,
) -> PDEGresult:
    """Run Algorithm 1 (T_w=0) or Algorithm 2 (T_w>0, then T production steps).

    Caches the corrected-pair gradient for the next prediction, so T_w+T
    iterations cost 1 + 2(T_w+T) full first-order oracle calls.

    OS samples J uniformly from {T_w, ..., T_w+T-1} and returns z_J.
    GS returns (x_{j+1}, y_{j+1}) for j = argmin S_t over that window.
    Pass ``os_index`` to skip sampling.
    """
    if project_x is None:
        project_x = project_unconstrained
    if project_y is None:
        project_y = project_unconstrained
    T = int(T)
    T_w = int(T_w)
    if T < 1:
        raise ValueError("T must be >= 1")
    if T_w < 0:
        raise ValueError("T_w must be >= 0")

    state = init_state(x0, y0)
    cache = (grad_x(state.x, state.y), grad_y(state.x, state.y))
    n_oracle = 1

    xs = [state.x]
    ys = [state.y]
    zs = [state.z]
    xis = [state.xi]
    ns = [state.n]
    vs = [state.v]
    Ss = []

    total = T_w + T
    for t in range(total):
        beta_t = 0.0 if t < T_w else params.beta
        state, S, cache, nq = step(
            state,
            grad_x,
            grad_y,
            params,
            project_x,
            project_y,
            beta=beta_t,
            cached_fgrad=cache,
        )
        n_oracle += nq
        xs.append(state.x)
        ys.append(state.y)
        zs.append(state.z)
        xis.append(state.xi)
        ns.append(state.n)
        vs.append(state.v)
        Ss.append(S)

    S_arr = jnp.stack(Ss)
    window = S_arr[T_w : T_w + T]
    gs_index = T_w + int(jnp.argmin(window))
    x_out = xs[gs_index + 1]
    y_out = ys[gs_index + 1]

    lo, hi = T_w, T_w + T
    if os_index is None:
        if key is not None:
            import jax

            os_index = int(jax.random.randint(key, (), lo, hi))
        else:
            os_index = int(np.random.randint(lo, hi))
    else:
        os_index = int(os_index)
        if not (lo <= os_index < hi):
            raise ValueError(f"os_index {os_index} not in [{lo}, {hi})")
    z_out = zs[os_index]

    return PDEGresult(
        state=state,
        S=S_arr,
        x_hist=jnp.stack(xs),
        y_hist=jnp.stack(ys),
        z_hist=jnp.stack(zs),
        xi_hist=jnp.stack(xis),
        n_hist=jnp.stack(ns),
        v_hist=jnp.stack(vs),
        x_out=x_out,
        y_out=y_out,
        z_out=z_out,
        gs_index=gs_index,
        os_index=os_index,
        n_oracle=n_oracle,
        T_w=T_w,
    )


def run_pdeg_warmup(
    grad_x: GradFn,
    grad_y: GradFn,
    x0,
    y0,
    params: PDEGparams,
    T: int,
    T_w: int | None = None,
    eta: float | None = None,
    H_mu_tau: float | None = None,
    project_x: ProjFn | None = None,
    project_y: ProjFn | None = None,
    key=None,
    os_index: int | None = None,
) -> PDEGresult:
    """Algorithm 2: fixed-center warm-up of length T_w, then T production steps.

    If ``T_w`` is omitted, it is computed from (4.31) using ``H_mu_tau`` and ``eta``.
    """
    if T_w is None:
        if eta is None or H_mu_tau is None:
            raise ValueError("Provide T_w, or both H_mu_tau and eta")
        T_w = warmup_length(params.beta, H_mu_tau, eta)
    return run_pdeg(
        grad_x,
        grad_y,
        x0,
        y0,
        params,
        project_x=project_x,
        project_y=project_y,
        T=T,
        T_w=int(T_w),
        key=key,
        os_index=os_index,
    )
