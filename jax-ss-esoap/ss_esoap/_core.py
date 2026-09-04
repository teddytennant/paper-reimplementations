"""Kronecker-factored preconditioning shared by SOAP and SS-ESOAP.

Equation and algorithm-line numbers refer to

    Wang, Toftrup, Loeschcke, Wang, Anandkumar,
    "SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed
    Learning", arXiv:2608.29448v1.

The pieces implemented here are Algorithm 2 (full pseudocode, Appendix B),
equations (1), (3), (4), (5) and (6), and the transition rule from Section 3.2.

Precision note. Every matmul in this module goes through ``_mm``, which pins
``jax.lax.Precision.HIGHEST``. On an H200 a float32 matmul defaults to TF32,
which keeps roughly 10 mantissa bits. That is not enough for the Kronecker
factor accumulation in (1), for the off-diagonal ratio (4), which is a
difference of two nearly equal Frobenius norms once the basis is good, or for
the rotations feeding the secant ratio (6), where the numerator is a
cancellation-prone trace. Pinning HIGHEST costs a few percent of step time and
makes the CPU and GPU trajectories agree.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Precision = jax.lax.Precision


def _mm(a: jax.Array, b: jax.Array) -> jax.Array:
    """Matmul that never falls back to TF32. See the module docstring."""
    return jax.lax.dot_general(
        a, b, (((a.ndim - 1,), (0,)), ((), ())), precision=Precision.HIGHEST
    )


def to_matrix(g: jax.Array) -> jax.Array:
    """Fold a parameter of rank >= 2 into the ``m x n`` matrix the paper assumes.

    Algorithm 2 is stated for ``W in R^{m x n}``. The paper never says what to do
    with convolution kernels or bias vectors, so this is our choice: collapse all
    leading axes into the row index and keep the last axis as the column index.
    Rank 0 and rank 1 leaves are not folded at all; see ``leaf_is_matrix``.
    """
    return g.reshape(-1, g.shape[-1])


def leaf_is_matrix(shape: tuple[int, ...], max_dim: int) -> bool:
    """Whether a leaf gets Kronecker preconditioning rather than plain Adam.

    Rank <= 1 leaves have no second Kronecker factor to speak of, and a leaf whose
    folded side exceeds ``max_dim`` would pay an O(d^3) eigendecomposition that
    Appendix F prices out of range. Both fall back to Adam, which is what
    Algorithm 2 degenerates to when ``Q_L = I`` and ``Q_R = I``.
    """
    if len(shape) < 2:
        return False
    m, n = math.prod(shape[:-1]), shape[-1]
    return m <= max_dim and n <= max_dim


def offdiag_ratio(a: jax.Array, q: jax.Array, eps: float) -> jax.Array:
    """Equation (4) / Algorithm 2 step 2: relative off-diagonal mass of ``A`` in ``Q``.

    ``rho(A, Q) = ||Q^T A Q - diag(Q^T A Q)||_F / (||Q^T A Q||_F + eps)``.

    Zero when ``Q`` diagonalizes ``A`` exactly, one when the basis carries no
    diagonal mass at all. The ``+ eps`` in the denominator is from Algorithm 2,
    not from equation (4) in the body, which omits it.

    The off-diagonal part is formed explicitly rather than as
    ``sqrt(||P||_F^2 - sum(diag(P)^2))``. Those two quantities are nearly equal
    once the basis is good, and the subtraction loses every digit that matters:
    the identity form returns about 1e-8 for a freshly computed eigenbasis in
    float64, where the true value is around 1e-15. That error alone is enough to
    change how often the trigger fires.
    """
    p = _mm(_mm(q.T, a), q)
    off = p - jnp.diag(jnp.diagonal(p))
    return jnp.linalg.norm(off) / (jnp.linalg.norm(p) + eps)


def gamma_from_rho(rho: jax.Array) -> jax.Array:
    """Algorithm 2 step 2 and Table 6: the variance downscaling ladder.

    ``gamma = 0.25`` if ``rho > 0.8``, ``0.5`` if ``rho > 0.5``, else ``0.75``.
    A worse-aligned basis forgets more of the accumulated second moment.
    """
    return jnp.where(rho > 0.8, 0.25, jnp.where(rho > 0.5, 0.5, 0.75))


def eigh_basis(a: jax.Array) -> jax.Array:
    """Eigenvectors of a symmetric Kronecker factor, columns sorted by decreasing
    eigenvalue.

    ``jnp.linalg.eigh`` returns ascending order. Sorting descending is not
    required by Algorithm 2, it just makes the basis a deterministic function of
    ``A`` so that a stale basis and a fresh one on the same factor line up.
    """
    _, vecs = jnp.linalg.eigh(_sym(a))
    return vecs[:, ::-1]


def _sym(a: jax.Array) -> jax.Array:
    return 0.5 * (a + a.T)


def secant_scale(
    s: jax.Array,
    y: jax.Array,
    ql: jax.Array,
    qr: jax.Array,
    dl: jax.Array,
    dr: jax.Array,
    tau_min: float,
    eps: float,
) -> jax.Array:
    """Equations (3), (5) and (6) / Algorithm 2 step 4: the self-scaling factor.

    With ``S = W_{t-1} - W_{t-2}``, ``Y = G_t - G_{t-1}`` and the Kronecker metric
    ``K`` whose quadratic form is ``s^T K s = Tr(S^T L^-1 S R^-1)``, Theorem 4.2
    gives

        tau = min(1, s^T y / (s^T K s)),

    the unique minimizer over ``0 < tau <= 1`` of the log-secant objective
    ``[log(s^T y / s^T (tau K) s)]^2``. Whenever the clip is inactive,
    ``s^T (tau K) s = s^T H s + s^T r``, so the scaled Kronecker secant energy
    equals the true directional curvature up to the directional secant error.

    We evaluate it in the eigenbasis, equations (5) and (6), so ``L^-1`` and
    ``R^-1`` are never formed:

        S~ = Q_L^T S Q_R,  Y~ = Q_L^T Y Q_R,
        c  = sum_ij Y~_ij S~_ij,
        a  = sum_ij S~_ij^2 / (D_L,i D_R,j).

    Algorithm 2 also floors the result at ``tau_min`` and falls back to 1 when
    the curvature condition ``c > 0`` fails.
    """
    st = _mm(_mm(ql.T, s), qr)
    yt = _mm(_mm(ql.T, y), qr)
    c = jnp.sum(yt * st)
    denom = jnp.maximum(dl, eps)[:, None] * jnp.maximum(dr, eps)[None, :]
    a = jnp.sum(st**2 / denom)
    raw = jnp.clip(c / jnp.where(a > 0, a, 1.0), tau_min, 1.0)
    return jnp.where((c > 0) & (a > 0), raw, 1.0)


def kron_quadratic_form(s: jax.Array, ell: jax.Array, r: jax.Array) -> jax.Array:
    """``s^T K s = Tr(S^T L^-1 S R^-1)``, Theorem 4.2, formed densely.

    Only used as a reference for the eigenspace shortcut in (5). It solves rather
    than inverting, and the solves run in whatever dtype ``s`` is in.
    """
    return jnp.trace(_mm(_mm(s.T, jnp.linalg.solve(ell, s)), jnp.linalg.inv(r)))


class LeafState(NamedTuple):
    """Per-parameter optimizer state. Matrix leaves carry the Kronecker factors;
    Adam-only leaves carry zero-sized stand-ins so the pytree stays uniform."""

    mu: jax.Array  # M~_t, momentum in the current eigenbasis
    nu: jax.Array  # V_t, coordinatewise second moment in the current eigenbasis
    left: jax.Array  # L_t, equation (1)
    right: jax.Array  # R_t, equation (1)
    ql: jax.Array  # Q_L
    qr: jax.Array  # Q_R
    prev_params: jax.Array  # W_{t-2} once t >= 2
    prev_grad: jax.Array  # G_{t-1}


class SSESOAPState(NamedTuple):
    count: jax.Array
    leaves: Any


def _empty(dtype) -> jax.Array:
    return jnp.zeros((0, 0), dtype=dtype)


def init_leaf(p: jax.Array, eps: float, max_dim: int) -> LeafState:
    """Algorithm 2 initialization: ``M~_0 = V_0 = 0``, ``L_0 = eps I``,
    ``R_0 = eps I``, ``Q_L = I``, ``Q_R = I``."""
    z = jnp.zeros_like(p)
    if not leaf_is_matrix(p.shape, max_dim):
        e = _empty(p.dtype)
        return LeafState(z, z, e, e, e, e, z, z)
    m, n = to_matrix(p).shape
    return LeafState(
        mu=jnp.zeros((m, n), p.dtype),
        nu=jnp.zeros((m, n), p.dtype),
        left=eps * jnp.eye(m, dtype=p.dtype),
        right=eps * jnp.eye(n, dtype=p.dtype),
        ql=jnp.eye(m, dtype=p.dtype),
        qr=jnp.eye(n, dtype=p.dtype),
        prev_params=z,
        prev_grad=z,
    )


class Config(NamedTuple):
    """Hyperparameters. Defaults are Table 6 where the paper states a value."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    tau_min: float = 0.1
    check_interval: int = 1
    trigger: float = 0.2
    warmup_steps: int = 0
    self_scale: bool = True
    adaptive_basis: bool = True
    precondition_frequency: int = 10
    variance_transition: str = "downscale"
    gamma: float | None = None
    basis_updates: bool = True
    max_precond_dim: int = 8192


def _rotate(x: jax.Array, ql_old: jax.Array, ql_new: jax.Array,
            qr_old: jax.Array, qr_new: jax.Array) -> jax.Array:
    """``X <- (Q_new^T Q_old) X (Q_old^T Q_new)``, Algorithm 2 step 2.

    Applied to the momentum so it keeps meaning the same parameter-space
    direction. SOAP applies it to the second moment as well; SS-ESOAP does not.
    """
    return _mm(_mm(_mm(ql_new.T, ql_old), x), _mm(qr_old.T, qr_new))


def update_leaf(
    grad: jax.Array,
    state: LeafState,
    param: jax.Array,
    count: jax.Array,
    cfg: Config,
) -> tuple[jax.Array, LeafState]:
    """One step of Algorithm 2 for a single parameter leaf.

    Returns the preconditioned direction ``U_t`` (step 5, before the learning
    rate and before the sign flip that optax applies), and the new state.
    """
    shape = param.shape
    if not leaf_is_matrix(shape, cfg.max_precond_dim):
        return _adam_leaf(grad, state, count, cfg)

    g = to_matrix(grad)

    # Step 1, equation (1): Kronecker factors.
    left = cfg.b2 * state.left + (1.0 - cfg.b2) * _mm(g, g.T)
    right = cfg.b2 * state.right + (1.0 - cfg.b2) * _mm(g.T, g)

    ql, qr, mu, nu = state.ql, state.qr, state.mu, state.nu

    # Step 2: basis update, adaptive (SS-ESOAP) or on a fixed period (SOAP).
    if cfg.basis_updates:
        if cfg.adaptive_basis:
            gate = (count > cfg.warmup_steps) & (
                jnp.mod(count, cfg.check_interval) == 0
            )
            rho = jnp.where(
                gate,
                jnp.maximum(
                    offdiag_ratio(left, ql, cfg.eps),
                    offdiag_ratio(right, qr, cfg.eps),
                ),
                0.0,
            )
            fire = gate & (rho > cfg.trigger)
        else:
            rho = jnp.zeros((), left.dtype)
            fire = jnp.mod(count, cfg.precondition_frequency) == 0

        def do_update(operands):
            ql_o, qr_o, mu_o, nu_o, rho_o = operands
            ql_n, qr_n = eigh_basis(left), eigh_basis(right)
            mu_n = _rotate(mu_o, ql_o, ql_n, qr_o, qr_n)
            if cfg.variance_transition == "downscale":
                g_t = (
                    gamma_from_rho(rho_o) if cfg.gamma is None
                    else jnp.asarray(cfg.gamma, nu_o.dtype)
                )
                nu_n = g_t.astype(nu_o.dtype) * nu_o
            elif cfg.variance_transition == "reproject":
                # Section 2.4's V^new = (Q_L^new)^T Q_L V Q_R^T Q_R^new. For a
                # matrix-shaped second moment that similarity is not positivity
                # preserving, and the negative entries reach a sqrt, so clamp at
                # zero. Without the clamp the SOAP baseline goes to NaN within a
                # few hundred PINN steps.
                nu_n = jnp.maximum(_rotate(nu_o, ql_o, ql_n, qr_o, qr_n), 0.0)
            elif cfg.variance_transition == "reset":
                nu_n = jnp.zeros_like(nu_o)
            elif cfg.variance_transition == "keep":
                nu_n = nu_o
            else:
                raise ValueError(cfg.variance_transition)
            return ql_n, qr_n, mu_n, nu_n

        ql, qr, mu, nu = jax.lax.cond(
            fire,
            do_update,
            lambda o: (o[0], o[1], o[2], o[3]),
            (ql, qr, mu, nu, rho),
        )

    # Step 3: Adam statistics in the eigenbasis.
    gt = _mm(_mm(ql.T, g), qr)
    mu = cfg.b1 * mu + (1.0 - cfg.b1) * gt
    nu = cfg.b2 * nu + (1.0 - cfg.b2) * gt**2

    # Step 4: self-scaling.
    if cfg.self_scale:
        s = to_matrix(param - state.prev_params)
        y = g - to_matrix(state.prev_grad)
        dl = jnp.diagonal(_mm(_mm(ql.T, left), ql))
        dr = jnp.diagonal(_mm(_mm(qr.T, right), qr))
        tau = jnp.where(
            count >= 2,
            secant_scale(s, y, ql, qr, dl, dr, cfg.tau_min, cfg.eps),
            1.0,
        )
    else:
        tau = jnp.ones((), g.dtype)

    # Step 5: bias-corrected Adam step in the eigenbasis, scaled by tau^{-1/2},
    # rotated back to parameter space.
    t = count.astype(g.dtype) + 1.0
    mu_hat = mu / (1.0 - cfg.b1**t)
    nu_hat = nu / (1.0 - cfg.b2**t)
    u_eig = tau ** (-0.5) * mu_hat / (jnp.sqrt(nu_hat) + cfg.eps)
    u = _mm(_mm(ql, u_eig), qr.T).reshape(shape)

    new = LeafState(
        mu=mu, nu=nu, left=left, right=right, ql=ql, qr=qr,
        prev_params=param, prev_grad=grad,
    )
    return u, new


def _adam_leaf(grad, state, count, cfg) -> tuple[jax.Array, LeafState]:
    """Algorithm 2 with ``Q_L = Q_R = I`` and ``tau = 1``, which is Adam."""
    mu = cfg.b1 * state.mu + (1.0 - cfg.b1) * grad
    nu = cfg.b2 * state.nu + (1.0 - cfg.b2) * grad**2
    t = count.astype(grad.dtype) + 1.0
    u = (mu / (1.0 - cfg.b1**t)) / (jnp.sqrt(nu / (1.0 - cfg.b2**t)) + cfg.eps)
    return u, state._replace(mu=mu, nu=nu, prev_grad=grad)
