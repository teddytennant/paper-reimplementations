"""Lazy second-order convex optimization steps.

N and S default to the formulas below. Pass N or S to override them.
The segment-search bisection stops after BISECTION_CAP steps.
"""

from __future__ import annotations

import math
import os
from typing import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array
GradFn = Callable[[Array], Array]
HessFn = Callable[[Array], Array]

BISECTION_CAP = 64


def _vec(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64).reshape(-1)


def _sym(Hmat) -> Array:
    Hmat = jnp.asarray(Hmat, dtype=jnp.float64)
    return 0.5 * (Hmat + jnp.swapaxes(Hmat, -1, -2))


def _ceil_snap(value: float) -> int:
    """Ceil in real arithmetic, snapping values that are an integer up to roundoff."""
    nearest = round(value)
    if abs(value - nearest) <= 1e-8 * max(1.0, abs(value)):
        return int(nearest)
    return int(math.ceil(value))


def extrapolation_params(kappa: float) -> tuple[float, int, float]:
    """Return (lambda, J, c) for the extrapolated objective."""
    kappa = float(kappa)
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    ratio = math.log(max(kappa, 1.0)) / (3.0 * math.log(64.0))
    J = max(1, _ceil_snap(ratio))
    lam = 1.0 / 64.0
    c = 1.0 + 1.0 / J
    return lam, J, c


def pagd_iteration_count(H_reg: float, gamma: float) -> int:
    """Default PAGD-Seg length N = ceil(64 * (J + 1)^(2/3) * (H_reg / gamma)^(1/3))."""
    gamma = float(gamma)
    H_reg = float(H_reg)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    kappa = H_reg / gamma
    _, J, _ = extrapolation_params(kappa)
    raw = 64.0 * (J + 1) ** (2.0 / 3.0) * (kappa ** (1.0 / 3.0))
    return _ceil_snap(raw)


def proximal_epoch_count(H_reg: float, gamma: float) -> int:
    """Default Cubic-Proximal epoch count S = ceil(log(kappa)), natural log."""
    gamma = float(gamma)
    H_reg = float(H_reg)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    kappa = H_reg / gamma
    if kappa <= 1.0:
        return 0
    return _ceil_snap(math.log(kappa))


def lazy_iteration_cost(n_grad: int, n_hess: int, m: int) -> int:
    """Accounted m-lazy cost: each Hessian query costs m gradient queries."""
    return int(n_grad) + int(m) * int(n_hess)


class CountingOracle:
    """Scalar objective with one counter tick per gradient or Hessian query."""

    def __init__(self, f, grad=None, hess=None):
        self.f = f
        self.n_grad = 0
        self.n_hess = 0
        self._grad = jax.jit(jax.grad(f)) if grad is None else grad
        self._hess = jax.jit(jax.hessian(f)) if hess is None else hess

    def value(self, x):
        return self.f(_vec(x))

    def grad(self, x):
        self.n_grad += 1
        return self._grad(_vec(x))

    def hess(self, x):
        self.n_hess += 1
        return self._hess(_vec(x))


def cubic_gradient(d, gamma: float) -> Array:
    """Gradient of (gamma / 3) * ||d||^3. Zero at the origin."""
    d = _vec(d)
    radius = jnp.linalg.norm(d)
    scale = jnp.where(radius > 0.0, float(gamma) * radius, 0.0)
    return scale * d


def cubic_hessian(d, gamma: float) -> Array:
    """Hessian of (gamma / 3) * ||d||^3. Zero at the origin."""
    d = _vec(d)
    n = int(d.shape[0])
    radius = jnp.linalg.norm(d)
    eye = jnp.eye(n, dtype=jnp.float64)
    safe = jnp.maximum(radius, 1e-30)
    dense = float(gamma) * (radius * eye + jnp.outer(d, d) / safe)
    zeros = jnp.zeros((n, n), dtype=jnp.float64)
    return jnp.where(radius > 0.0, dense, zeros)


def proximal_gradient(y, x, gamma: float, grad_f: GradFn) -> Array:
    y = _vec(y)
    x = _vec(x)
    return _vec(grad_f(y)) + cubic_gradient(y - x, gamma)


def proximal_hessian(y, x, gamma: float, hess_f: HessFn) -> Array:
    y = _vec(y)
    x = _vec(x)
    return _sym(hess_f(y)) + cubic_hessian(y - x, gamma)


def crn_step(z, g, Hmat, H_reg: float) -> Array:
    """Cubic-regularized Newton step C_H(z) = z + s*.

    s* minimizes <g, s> + (1/2) s^T Hmat s + (H_reg / 3) ||s||^3.
    The stationarity condition is (Hmat + lambda I) s = -g with lambda = H_reg * ||s||.
    Solved by jnp.linalg.eigh and a 1D search on lambda. If the lambda = 0 Newton
    step already satisfies H_reg * ||s|| = 0, that step is used.
    """
    z = _vec(z)
    g = _vec(g)
    Hmat = _sym(Hmat)
    H_reg = float(H_reg)
    if H_reg < 0.0:
        raise ValueError("H_reg must be nonnegative")
    if Hmat.shape != (g.shape[0], g.shape[0]):
        raise ValueError("Hmat shape does not match g")

    evals, evecs = jnp.linalg.eigh(Hmat)
    evals_h = np.asarray(evals, dtype=np.float64)
    evecs_h = np.asarray(evecs, dtype=np.float64)
    g_hat = evecs_h.T @ np.asarray(g, dtype=np.float64)

    def recover_s(lam: float) -> np.ndarray:
        denom = evals_h + lam
        if lam == 0.0:
            s_hat = np.zeros_like(g_hat)
            mask = np.abs(denom) > 1e-12
            s_hat[mask] = -g_hat[mask] / denom[mask]
            return evecs_h @ s_hat
        return evecs_h @ (-g_hat / denom)

    gnorm = float(np.linalg.norm(g_hat))
    if gnorm == 0.0 or H_reg == 0.0:
        return z + jnp.asarray(recover_s(0.0), dtype=jnp.float64)

    mu_min = float(evals_h[0])
    psd = mu_min >= -1e-12
    if psd:
        s0 = recover_s(0.0)
        if H_reg * float(np.linalg.norm(s0)) == 0.0:
            return z + jnp.asarray(s0, dtype=jnp.float64)

    lam_floor = 0.0 if psd else -mu_min
    mu_eff = 0.0 if psd else mu_min
    disc = mu_eff * mu_eff + 4.0 * H_reg * gnorm
    lam_hi = 0.5 * (-mu_eff + math.sqrt(max(disc, 0.0)))
    lam_hi = max(lam_hi, lam_floor + 1e-12)

    def sigma(lam: float) -> float:
        denom = evals_h + lam
        if np.min(np.abs(denom)) < 1e-18:
            return math.inf
        return float(np.linalg.norm(g_hat / denom))

    for _ in range(8):
        if sigma(lam_hi) <= lam_hi / H_reg:
            break
        lam_hi = max(2.0 * lam_hi, lam_hi + 1.0)

    lo = lam_floor
    hi = lam_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sigma(mid) > mid / H_reg:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)

    for _ in range(12):
        denom = evals_h + lam
        if np.min(np.abs(denom)) < 1e-18:
            break
        inv = 1.0 / denom
        sigma_val = float(np.linalg.norm(g_hat * inv))
        if sigma_val == 0.0:
            break
        sigma_prime = -float(np.sum((g_hat**2) * (inv**3))) / sigma_val
        phi = sigma_val - lam / H_reg
        phi_prime = sigma_prime - 1.0 / H_reg
        if phi_prime == 0.0 or not math.isfinite(phi_prime):
            break
        step = phi / phi_prime
        lam_new = lam - step
        if lam_new <= lam_floor or not math.isfinite(lam_new):
            break
        lam = lam_new
        if abs(step) <= 1e-14 * (1.0 + abs(lam)):
            break

    return z + jnp.asarray(recover_s(lam), dtype=jnp.float64)


def cubic_regularized_newton(z, x, gamma: float, L: float, grad_f: GradFn, hess_f: HessFn) -> Array:
    """One cubic-regularized Newton step on the proximal function at center x."""
    gamma = float(gamma)
    H_reg = float(L) + 2.0 * gamma
    g = proximal_gradient(z, x, gamma, grad_f)
    Hmat = proximal_hessian(z, x, gamma, hess_f)
    return crn_step(z, g, Hmat, H_reg)


def finite_seg_search(grad_g: GradFn, x, v, zeta: float, max_steps: int = BISECTION_CAP):
    """Bisection on the directional derivative of g along the segment [x, v].

    phi(beta) = g(x + beta (v - x)). On a normal exit with the root inside the
    segment, -zeta <= phi'(l) <= 0. The loop stops after max_steps bisections
    and returns the current left endpoint.
    """
    x = _vec(x)
    v = _vec(v)
    direction = v - x
    zeta = float(zeta)
    max_steps = int(max_steps)

    def phi_prime(beta: float) -> float:
        point = x + beta * direction
        return float(jnp.dot(_vec(grad_g(point)), direction))

    slope_left = phi_prime(0.0)
    if slope_left >= 0.0:
        return x
    if phi_prime(1.0) <= 0.0:
        return v

    left = 0.0
    right = 1.0
    steps = 0
    while -slope_left > zeta and steps < max_steps:
        steps += 1
        mid = 0.5 * (left + right)
        slope = phi_prime(mid)
        if slope == 0.0:
            return x + mid * direction
        if slope < 0.0:
            left = mid
            slope_left = slope
        else:
            right = mid
        if right - left == 0.0:
            break
    return x + left * direction


def _curvature_shift(H_reg: float, Delta: float) -> float:
    if Delta <= 0.0:
        return 0.0
    return (H_reg * H_reg * Delta) ** (1.0 / 3.0)


def extrapolated_gradient(v, y, prox_grad: GradFn, lam: float, c: float, J: int) -> Array:
    """Gradient of sum_{i=0}^J lambda^i f(y + c^i (v - y))."""
    v = _vec(v)
    y = _vec(y)
    acc = jnp.zeros_like(v)
    diff = v - y
    for i in range(int(J) + 1):
        scale = c**i
        weight = (lam**i) * scale
        acc = acc + weight * _vec(prox_grad(y + scale * diff))
    return acc


def pagd_seg(
    prox_grad: GradFn,
    prox_hess: HessFn,
    y,
    Delta: float,
    H_reg: float,
    gamma: float,
    N: int | None = None,
):
    """Preconditioned AGD with one segment search per iteration.

    The Hessian of the proximal function is queried once, at y.
    N overrides the default iteration count. The default is not reduced.
    """
    y = _vec(y)
    Delta = float(Delta)
    H_reg = float(H_reg)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if N is None:
        N = pagd_iteration_count(H_reg, gamma)
    N = int(N)
    if N < 0:
        raise ValueError("N must be nonnegative")

    Hmat = _sym(prox_hess(y))
    kappa = H_reg / gamma
    lam, J, c = extrapolation_params(kappa)
    shift = _curvature_shift(H_reg, Delta)
    n = int(y.shape[0])
    coeff = 100.0 * ((J + 1) ** (4.0 / 3.0)) * shift
    B = 2.0 * (J + 1) * Hmat + coeff * jnp.eye(n, dtype=jnp.float64)

    def grad_g(point):
        return extrapolated_gradient(point, y, prox_grad, lam, c, J)

    zeta = Delta / 100.0
    x_k = y
    v_k = y
    A = 0.0
    for _ in range(N):
        y_k = finite_seg_search(grad_g, x_k, v_k, zeta)
        h_k = grad_g(y_k)
        a = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * A))
        A = A + a
        step = jnp.linalg.solve(B, h_k)
        x_k = y_k - step
        v_k = v_k - a * step
    return x_k


def cubic_proximal(
    grad_f: GradFn,
    hess_f: HessFn,
    x,
    gamma: float,
    L: float,
    N: int | None = None,
    S: int | None = None,
):
    """Approximate cubic proximal point of f at x.

    N overrides the PAGD-Seg iteration count. S overrides the epoch count
    S = ceil(log(kappa)). Defaults are the formulas. If the initial gap
    estimate is 0, the epoch loop is skipped and C(y_0) is returned.
    """
    x = _vec(x)
    gamma = float(gamma)
    L = float(L)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    H_reg = L + 2.0 * gamma

    def prox_grad(y):
        return proximal_gradient(y, x, gamma, grad_f)

    def prox_hess(y):
        return proximal_hessian(y, x, gamma, hess_f)

    def C(z):
        return crn_step(z, prox_grad(z), prox_hess(z), H_reg)

    z0 = C(x)
    y = C(z0)
    grad_norm = float(jnp.linalg.norm(prox_grad(y)))
    Delta = (2.0 * math.sqrt(2.0) / (3.0 * math.sqrt(gamma))) * (grad_norm**1.5)
    if Delta == 0.0:
        return C(y)

    if S is None:
        S = proximal_epoch_count(H_reg, gamma)
    S = int(S)
    if S < 0:
        raise ValueError("S must be nonnegative")
    for _ in range(S):
        y = pagd_seg(prox_grad, prox_hess, y, Delta, H_reg, gamma, N=N)
        Delta = 0.75 * Delta
    return C(y)


def search_free_ms_acceleration(
    grad_f: GradFn,
    hess_f: HessFn,
    x0,
    gamma: float,
    T: int,
    L: float,
    N: int | None = None,
    S: int | None = None,
):
    """Search-free Monteiro-Svaiter acceleration.

    y_0 is the cubic proximal of x_0, computed before the loop. Later proximal
    points are recomputed only when t > 0, from the current x_t. The momentum
    update uses grad f, not the proximal gradient. T = 0 returns y_0. The
    returned point is the latest proximal output (y_0 when T <= 1).
    """
    x = _vec(x0)
    v = x
    A = 0.0
    y = cubic_proximal(grad_f, hess_f, x, gamma, L, N=N, S=S)
    for t in range(int(T)):
        if t > 0:
            y = cubic_proximal(grad_f, hess_f, x, gamma, L, N=N, S=S)
        a = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * A))
        A_next = A + a
        x = (A * y + a * v) / A_next
        v = v - a * _vec(grad_f(y))
        A = A_next
    return y
