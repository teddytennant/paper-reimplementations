"""Binary KL, inverse KL, and DIS-R / DIS-V certificates (Eqs 14-15)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_dis_smv.dirichlet import dirichlet_renyi, log_beta, log_density_ratio


def binary_kl(p: jax.Array, q: jax.Array) -> jax.Array:
    """Bernoulli KL kl(p || q) = p ln(p/q) + (1-p) ln((1-p)/(1-q)), with 0 log 0 = 0."""
    p = jnp.asarray(p)
    q = jnp.asarray(q)
    dtype = jnp.result_type(p, q, jnp.float32)
    p = p.astype(dtype)
    q = q.astype(dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)

    def term(a, b):
        # a ln(a/b); 0 ln(0/b) = 0; a>0 and b=0 -> inf
        return jnp.where(
            a == 0.0,
            zero,
            jnp.where(b == 0.0, inf, a * jnp.log(a / b)),
        )

    return term(p, q) + term(1.0 - p, 1.0 - q)


def _bisect_kl_inv(q: jax.Array, eps: jax.Array, steps: int = 52) -> jax.Array:
    """Max p in [q, 1] with kl(q || p) <= eps, by bisection on p."""
    lo = q
    hi = jnp.maximum(jnp.asarray(1.0 - 1.0e-6, dtype=q.dtype), lo)

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        too_big = binary_kl(q, mid) > eps
        hi_n = jnp.where(too_big, mid, hi)
        lo_n = jnp.where(too_big, lo, mid)
        return lo_n, hi_n

    lo, hi = jax.lax.fori_loop(0, steps, body, (lo, hi))
    return 0.5 * (lo + hi)


@jax.custom_jvp
def kl_inv(q: jax.Array, eps: jax.Array) -> jax.Array:
    """kl^{-1}(q || eps) = max {p in [q, 1] : kl(q || p) <= eps} (Appendix B.1).

    If eps <= 0 the unique feasible point is p = q (kl is nonnegative).
    """
    q = jnp.asarray(q)
    eps = jnp.asarray(eps, dtype=q.dtype)
    q = jnp.clip(q, 0.0, 1.0)

    def at_boundary(_):
        return q

    def interior(_):
        return _bisect_kl_inv(q, eps)

    return jax.lax.cond((q >= 1.0 - 1.0e-12) | (eps <= 0.0), at_boundary, interior, q)


@kl_inv.defjvp
def _kl_inv_jvp(primals, tangents):
    q, eps = primals
    q_dot, eps_dot = tangents
    p = kl_inv(q, eps)
    qe = jnp.clip(q, 1.0e-12, 1.0 - 1.0e-12)
    pe = jnp.clip(p, 1.0e-12, 1.0 - 1.0e-12)
    dkl_dp = -qe / pe + (1.0 - qe) / (1.0 - pe)
    dkl_dq = jnp.log(qe / pe) - jnp.log((1.0 - qe) / (1.0 - pe))
    denom = jnp.where(jnp.abs(dkl_dp) < 1.0e-8, jnp.ones_like(dkl_dp), dkl_dp)
    newton = (eps_dot - dkl_dq * q_dot) / denom
    at_q = (eps <= 0.0) | (p <= q + 1.0e-7)
    p_dot = jnp.where(at_q, q_dot, newton)
    return p, p_dot


def dis_r_penalty(
    alpha: jax.Array,
    beta: jax.Array,
    rho: jax.Array,
    n: int | jax.Array,
    delta: float,
) -> jax.Array:
    """DIS-R penalty of Corollary 1 / Eq 14, the right-hand side of the kl.

    (1/n) [ ln B(beta)/B(alpha) + sum_j (alpha_j - beta_j) ln(rho_j)
            + ln(2 sqrt(n) / delta) ]
    """
    n_f = jnp.asarray(n, dtype=jnp.asarray(alpha).dtype)
    log_ratio = log_density_ratio(rho, alpha, beta)
    extra = jnp.log(2.0 * jnp.sqrt(n_f) / delta)
    return (log_ratio + extra) / n_f


def dis_v_penalty(
    alpha: jax.Array,
    beta: jax.Array,
    n: int | jax.Array,
    delta: float,
    lam: float,
) -> jax.Array:
    """DIS-V penalty of Corollary 2 / Theorem 7 / Eq 15.

    (1/n) * [ ((2 lam - 1)/(lam - 1)) ln(2/delta)
              + ln B(beta)/B(alpha)
              + 1/(lam-1) ln B(lam alpha + (1-lam) beta)/B(alpha)
              + ln(2 sqrt(n)) ]
    """
    n_f = jnp.asarray(n, dtype=jnp.asarray(alpha).dtype)
    dlam = dirichlet_renyi(alpha, beta, lam)
    term = (
        ((2.0 * lam - 1.0) / (lam - 1.0)) * jnp.log(2.0 / delta)
        + dlam
        + jnp.log(2.0 * jnp.sqrt(n_f))
    )
    return term / n_f


def dis_r_certificate(
    rhat: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    rho: jax.Array,
    n: int | jax.Array,
    delta: float,
) -> jax.Array:
    """DIS-R bound of Eq 14: kl^{-1}(Rhat_S(MVrho) || DIS-R penalty)."""
    eps = dis_r_penalty(alpha, beta, rho, n, delta)
    return kl_inv(rhat, eps)


def dis_v_certificate(
    rhat: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    n: int | jax.Array,
    delta: float,
    lam: float,
) -> jax.Array:
    """DIS-V bound of Eq 15: kl^{-1}(Rhat_S(MVrho) || DIS-V penalty)."""
    eps = dis_v_penalty(alpha, beta, n, delta, lam)
    return kl_inv(rhat, eps)


def log_beta_ratio(alpha: jax.Array, beta: jax.Array) -> jax.Array:
    """ln B(beta)/B(alpha), used by tests to expand Eqs 14-15."""
    return log_beta(beta) - log_beta(alpha)
