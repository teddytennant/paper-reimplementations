"""p=2 unconstrained tensor step, Anchored-Tensor-Method, and R-Adaptive-ATM.

Definition 2.2 with p=2 and X = R^d (normal cone {0}):

    0 = g + H s + (M/2) ||s|| s,   y = x + s,

where g = F(x) and H = DF(x). Equivalently (H + λ I) s = -g with
λ = (M/2) ||s||. H is not assumed symmetric; the secular equation is
solved by bisection and a linear solve, not an eigendecomposition.

a_p, m_p, theta_p are existential in the paper. The concrete values
below only set the initial trial lengths. Acceptance
(1/μ) ||F(u)|| ≤ δ is the gate (Condition 4.1, unconstrained).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Existential constants, fixed so the acceptance test can be checked.
M_P = 1.0
A_P = 0.5
THETA_P = 0.5

BISECTION_ITERS = 60
G_TOL = 1e-12
MODEL_TOL = 1e-8
MAX_TRIALS = 12
STEP_CAP = 40


def tensor_step(F, x, M, g=None, H=None):
    """One unconstrained p=2 tensor step of F at x with regularization M.

    If g and H are omitted they are F(x) and jax.jacfwd(F)(x). Returns y = x + s.
    If ||g|| < 1e-12, returns x.
    """
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    M = float(M)
    if M < 0:
        raise ValueError("M must be nonnegative")
    if g is None:
        if F is None:
            raise ValueError("tensor_step requires F or both g and H")
        g = F(x)
    g = jnp.asarray(g, dtype=jnp.float64).reshape(-1)
    if g.shape[0] != x.shape[0]:
        raise ValueError("g and x have different dimensions")
    if float(jnp.linalg.norm(g)) < G_TOL:
        return x
    if H is None:
        if F is None:
            raise ValueError("tensor_step requires F or both g and H")
        H = jax.jacfwd(F)(x)
    H = jnp.asarray(H, dtype=jnp.float64)
    if H.shape != (x.shape[0], x.shape[0]):
        raise ValueError("H must be a square Jacobian")
    s = _secular_solve(np.asarray(g), np.asarray(H), M)
    return x + jnp.asarray(s, dtype=jnp.float64)


def _secular_solve(g, H, M, n_iter=BISECTION_ITERS):
    """Bisection on λ > 0 for (H + λ I) s = -g, λ = (M/2) ||s||.

    φ(λ) = (M/2)||s(λ)|| - λ is positive for small λ and negative for large λ.
    """
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    H = np.asarray(H, dtype=np.float64)
    d = g.shape[0]
    eye = np.eye(d, dtype=np.float64)

    def s_of(lam):
        return -np.linalg.solve(H + lam * eye, g)

    def phi_and_s(lam):
        s = s_of(lam)
        return 0.5 * M * np.linalg.norm(s) - lam, s

    lam_hi = 1e-8
    phi_hi = None
    s_hi = None
    for _ in range(80):
        phi_hi, s_hi = phi_and_s(lam_hi)
        if np.isfinite(phi_hi) and phi_hi <= 0.0:
            break
        lam_hi *= 2.0
    else:
        raise RuntimeError("secular bracket failed: phi(lambda) stayed positive")

    lam_lo = min(1e-16, lam_hi * 1e-8)
    phi_lo, s_lo = phi_and_s(lam_lo)
    if not np.isfinite(phi_lo) or phi_lo <= 0.0:
        # Root is at or below lam_lo (or the tiny shift already overshot).
        s = s_lo if np.all(np.isfinite(s_lo)) else s_hi
        _check_model(g, H, s, M)
        return s

    for _ in range(n_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        phi_mid, _ = phi_and_s(lam_mid)
        if not np.isfinite(phi_mid):
            lam_lo = lam_mid
            continue
        if phi_mid > 0.0:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    lam = 0.5 * (lam_lo + lam_hi)
    s = s_of(lam)
    _check_model(g, H, s, M)
    return s


def _check_model(g, H, s, M):
    resid = g + H @ s + (0.5 * M) * np.linalg.norm(s) * s
    nrm = float(np.linalg.norm(resid))
    if not np.isfinite(nrm) or nrm >= MODEL_TOL:
        raise RuntimeError(f"tensor model residual {nrm} exceeds {MODEL_TOL}")


def anchored_tensor_method(F, x, mu, M, R, S, m_p=M_P):
    """Anchored-Tensor-Method (Algorithm 2) with R, S iteration counts.

    F is the target μ-strongly monotone operator (the outer G_t). Phase I
    runs R tensor steps with a decreasing extra anchor

        μ_k = μ (1 + m_p)^{R-1-k},
        z |-> F(z) + (μ_k - μ) (z - x),

    which is Algorithm 2's G_k. Phase II runs S tensor steps on F itself
    (the target resolvent operator). An increasing schedule that keeps the
    large anchor through Phase II does not make ||F(u)|| small.
    """
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    mu = float(mu)
    M = float(M)
    R = int(R)
    S = int(S)
    if R < 0 or S < 0:
        raise ValueError("R and S must be nonnegative")
    anchor = x
    z = x
    eye = jnp.eye(z.shape[0], dtype=jnp.float64)

    for k in range(R):
        mu_k = mu * (1.0 + m_p) ** (R - 1 - k)
        extra = mu_k - mu
        H = jax.jacfwd(F)(z) + extra * eye
        g = F(z) + extra * (z - anchor)
        z = tensor_step(F, z, M, g=g, H=H)

    for _ in range(S):
        z = tensor_step(F, z, M)

    return z


def _initial_lengths(mu, M, delta, m_p, a_p, theta_p, p):
    """S0 and ρ(μ) from the concrete constants. p=2 makes the 2 in the
    denominator agree with the paper's p * log(1/theta_p)."""
    if p < 1:
        raise ValueError("p must be a positive integer")
    h_p = (2.0 ** p) * (m_p + 1.0) * (a_p ** (p - 1)) / math.factorial(p)
    rho = (mu / (a_p * M)) ** (1.0 / (p - 1))
    ratio = max(h_p * rho / delta, 1.0)
    denom = 2.0 * math.log(1.0 / theta_p)
    S0 = max(1, math.ceil(math.log(ratio) / denom))
    return h_p, rho, S0


def _clamp_steps(R, S, cap):
    R = max(1, int(R))
    S = max(1, int(S))
    if R + S <= cap:
        return R, S
    S = min(S, cap - 1)
    R = min(R, cap - S)
    return max(1, R), max(1, S)


@dataclass
class ATMResult:
    u: jnp.ndarray
    n: jnp.ndarray
    accepted: bool
    trials: int
    scaled_residual: float


def r_adaptive_atm(
    F,
    x,
    mu,
    M,
    delta,
    *,
    m_p=M_P,
    a_p=A_P,
    theta_p=THETA_P,
    p=2,
    max_trials=MAX_TRIALS,
    step_cap=STEP_CAP,
    grow_S=False,
):
    """R-Adaptive-ATM (Algorithm 3), unconstrained.

    ρ(μ) = (μ / (a_p M))^{1/(p-1)}
    S0 = max(1, ceil(log(max(h_p ρ(μ)/δ, 1)) / (2 log(1/theta_p))))
    R_j = max(1, ceil(2^j S0 ρ(μ)))

    S_j = S0 as written in Algorithm 3 (grow_S=False). The acceptance test
    is the gate. n = 0 on R^d. Returns as soon as (1/μ) ||F(u)|| ≤ δ.
    """
    if p != 2:
        raise NotImplementedError("tensor step is implemented only for p=2")
    mu = float(mu)
    M = float(M)
    delta = float(delta)
    if mu <= 0 or M <= 0 or delta < 0:
        raise ValueError("mu, M must be positive and delta nonnegative")

    _, rho, S0 = _initial_lengths(mu, M, delta, m_p, a_p, theta_p, p)
    x = jnp.asarray(x, dtype=jnp.float64).reshape(-1)
    u = x
    n = jnp.zeros_like(u)
    scaled = float("inf")
    accepted = False
    trials = 0

    for j in range(max_trials):
        R_j = max(1, math.ceil((2 ** j) * S0 * rho))
        S_j = S0 * (j + 1) if grow_S else S0
        R_j, S_j = _clamp_steps(R_j, S_j, step_cap)
        u = anchored_tensor_method(F, x, mu, M, R_j, S_j, m_p=m_p)
        n = jnp.zeros_like(u)
        scaled = float(jnp.linalg.norm(F(u) + n)) / mu
        trials = j + 1
        if scaled <= delta:
            accepted = True
            break

    return ATMResult(u=u, n=n, accepted=accepted, trials=trials, scaled_residual=scaled)
