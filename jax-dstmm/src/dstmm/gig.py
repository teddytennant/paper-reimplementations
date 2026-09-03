"""Generalised inverse Gaussian law of Proposition 2, plus its inverse-gamma limit.

The paper's GIG parameterisation is equation (13),

    f(w; psi, chi, lam) = (psi/chi)^(lam/2) / (2 K_lam(sqrt(psi*chi)))
                          * w^(lam-1) * exp(-(psi*w + chi/w)/2),   w > 0,

and the inverse-gamma parameterisation is the shape-scale one of Section 2.1,

    f_IG(w; a, b) = b^a / Gamma(a) * w^(-a-1) * exp(-b/w).

Proposition 2 gives W | (y, S=s) ~ GIG(A_s, nu_s + D_js, -(nu_s+p)/2), with the
moment identity (16) and the log-moment (22), and the A_s = 0 limit (17)-(18).

Sampling uses rejection from a tangent hull. On t = log w the GIG log-density is
lam*t - (psi*e^t + chi*e^-t)/2, which is concave, so the pointwise minimum of a
few tangent lines is an exact upper envelope and it is piecewise exponential, so
it can be sampled by inverse CDF. Six tangent points placed at fixed multiples of
the curvature scale at the mode accept about 85% of proposals, and the loop
retries the rejected entries only.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from dstmm.bessel import log_bessel_k, dlog_bessel_k_dnu

# Tangent abscissae in units of the curvature scale at the mode. None sits at the
# mode itself: the slope there is exactly zero, and a zero-slope segment is a
# uniform rather than an exponential piece.
_TANGENT_OFFSETS = jnp.array([-3.5, -1.8, -0.6, 0.6, 1.8, 3.5])
_MAX_ROUNDS = 60


def gig_log_norm(psi, chi, lam):
    """log of the GIG normalising constant, log[2 K_lam(sqrt(psi*chi)) (chi/psi)^(lam/2)].

    This is log of the integral of w^(lam-1) exp(-(psi*w + chi/w)/2) over w > 0,
    the reciprocal of the leading factor in equation (13).
    """
    psi = jnp.asarray(psi, jnp.float64)
    chi = jnp.asarray(chi, jnp.float64)
    lam = jnp.asarray(lam, jnp.float64)
    return jnp.log(2.0) + 0.5 * lam * jnp.log(chi / psi) + log_bessel_k(lam, jnp.sqrt(psi * chi))


def gig_logpdf(w, psi, chi, lam):
    """log of equation (13)."""
    w = jnp.asarray(w, jnp.float64)
    return (lam - 1.0) * jnp.log(w) - 0.5 * (psi * w + chi / w) - gig_log_norm(psi, chi, lam)


def gig_moment(q, psi, chi, lam):
    """E(W^q) from equation (16): (chi/psi)^(q/2) K_(lam+q)(sqrt(psi*chi)) / K_lam(sqrt(psi*chi))."""
    arg = jnp.sqrt(psi * chi)
    return jnp.exp(
        0.5 * q * jnp.log(chi / psi) + log_bessel_k(lam + q, arg) - log_bessel_k(lam, arg)
    )


def gig_log_moment(psi, chi, lam):
    """E(log W) from equation (22): (1/2) log(chi/psi) + d/dlam log K_lam(sqrt(psi*chi))."""
    return 0.5 * jnp.log(chi / psi) + dlog_bessel_k_dnu(lam, jnp.sqrt(psi * chi))


def gig_mode(psi, chi, lam):
    """Mode of the GIG density, (lam + sqrt(lam^2 + psi*chi)) / psi.

    Written as chi / (sqrt(lam^2 + psi*chi) - lam) when lam < 0, which is the
    branch the model always takes since lam = -(nu+p)/2, to keep the difference
    of two nearly equal numbers out of the numerator.
    """
    root = jnp.sqrt(lam * lam + psi * chi)
    return jnp.where(lam >= 0.0, (lam + root) / psi, chi / jnp.maximum(root - lam, 1e-300))


def inverse_gamma_logpdf(w, a, b):
    """log f_IG(w; a, b) in the shape-scale parameterisation of Section 2.1."""
    w = jnp.asarray(w, jnp.float64)
    return a * jnp.log(b) - gammaln(a) - (a + 1.0) * jnp.log(w) - b / w


def sample_inverse_gamma(key, a, b, shape=None):
    """Draw from IG(a, b). If G ~ Gamma(a, 1) then b/G ~ IG(a, b)."""
    a = jnp.asarray(a, jnp.float64)
    b = jnp.asarray(b, jnp.float64)
    if shape is None:
        shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    g = jax.random.gamma(key, a, shape=shape, dtype=jnp.float64)
    return b / g


def _log_kernel(t, psi, chi, lam):
    """lam*t - (psi*e^t + chi*e^-t)/2, the log GIG density on t = log w up to a constant."""
    return lam * t - 0.5 * (psi * jnp.exp(t) + chi * jnp.exp(-t))


def _log_kernel_grad(t, psi, chi, lam):
    return lam - 0.5 * (psi * jnp.exp(t) - chi * jnp.exp(-t))


def sample_gig(key, psi, chi, lam, shape=None):
    """Draw from GIG(psi, chi, lam) as parameterised in equation (13).

    psi must be strictly positive. The A_s = 0 case of Proposition 2 is the
    inverse-gamma limit (17); use sample_inverse_gamma for it.
    """
    psi, chi, lam = (jnp.asarray(v, jnp.float64) for v in (psi, chi, lam))
    if shape is None:
        shape = jnp.broadcast_shapes(jnp.shape(psi), jnp.shape(chi), jnp.shape(lam))
    psi, chi, lam = (jnp.broadcast_to(v, shape) for v in (psi, chi, lam))

    t_mode = jnp.log(gig_mode(psi, chi, lam))
    # -h''(t_mode) = sqrt(lam^2 + psi*chi), so the curvature scale is its -1/2 power.
    scale = (lam * lam + psi * chi) ** -0.25
    tk = t_mode[..., None] + scale[..., None] * _TANGENT_OFFSETS  # (..., 6)

    pk, ck, lk = (v[..., None] for v in (psi, chi, lam))
    slope = _log_kernel_grad(tk, pk, ck, lk)  # decreasing in k
    intercept = _log_kernel(tk, pk, ck, lk) - tk * slope

    # Breakpoints of the piecewise-linear upper envelope min_k (a_k + b_k t).
    z = (intercept[..., 1:] - intercept[..., :-1]) / (slope[..., :-1] - slope[..., 1:])
    neg_inf = jnp.full(z.shape[:-1] + (1,), -jnp.inf)
    pos_inf = jnp.full(z.shape[:-1] + (1,), jnp.inf)
    lo = jnp.concatenate([neg_inf, z], axis=-1)
    hi = jnp.concatenate([z, pos_inf], axis=-1)

    rising = slope > 0.0
    anchor = jnp.where(rising, hi, lo)  # the finite end each segment decays away from
    other = jnp.where(rising, lo, hi)
    decay = jnp.exp(slope * (other - anchor))  # in [0, 1), exactly 0 at an infinite end
    log_seg_w = (
        intercept + slope * anchor + jnp.log1p(-decay) - jnp.log(jnp.abs(slope))
    )
    log_seg_w = log_seg_w - logsumexp(log_seg_w, axis=-1, keepdims=True)

    def round_body(state):
        it, key_, accepted, out = state
        key_, k1, k2, k3 = jax.random.split(key_, 4)
        idx = jax.random.categorical(k1, log_seg_w, axis=-1)
        take = lambda arr: jnp.take_along_axis(arr, idx[..., None], axis=-1)[..., 0]
        b, a, anc, dec = take(slope), take(intercept), take(anchor), take(decay)
        u = jax.random.uniform(k2, shape, dtype=jnp.float64, minval=1e-300, maxval=1.0)
        t = anc + jnp.log(dec + u * (1.0 - dec)) / b
        log_accept = _log_kernel(t, psi, chi, lam) - (a + b * t)
        v = jax.random.uniform(k3, shape, dtype=jnp.float64)
        fresh = (jnp.log(v) <= log_accept) & (~accepted)
        return it + 1, key_, accepted | fresh, jnp.where(fresh, t, out)

    def round_cond(state):
        it, _, accepted, _ = state
        return (it < _MAX_ROUNDS) & (~jnp.all(accepted))

    init = (0, key, jnp.zeros(shape, bool), t_mode)
    _, _, _, t_out = jax.lax.while_loop(round_cond, round_body, init)
    return jnp.exp(t_out)
