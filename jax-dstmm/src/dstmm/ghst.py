"""The generalised-hyperbolic skew-t component of Section 2.4.

Under equations (1)-(2),

    X | (W = w) ~ N_p(mu + w*alpha, w*Sigma),   W ~ IG(nu/2, nu/2),

and the paper writes X ~ GHSt_p(mu, Sigma, alpha, nu). With

    d^2(y; mu, Sigma) = (y-mu)' Sigma^-1 (y-mu),   q(alpha; Sigma) = alpha' Sigma^-1 alpha,

the density in Section 2.4 is

    g(y) = nu^(nu/2) / [(2 pi)^(p/2) |Sigma|^(1/2) Gamma(nu/2) 2^(nu/2 - 1)]
           * exp{(y-mu)' Sigma^-1 alpha}
           * [(nu + d^2) / q]^(-(nu+p)/4)
           * K_((nu+p)/2)( sqrt( q * (nu + d^2) ) ).

Section 2.4 states the alpha -> 0 limit is the multivariate Student-t density,
and Section 3.2 says the implementation switches to the Student-t form when
q(alpha; Sigma) is numerically negligible rather than evaluating the Bessel form
at a near-degenerate skewness. That switch is _Q_FLOOR below.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from dstmm.bessel import log_bessel_k
from dstmm.gig import sample_inverse_gamma

# Below this value of q(alpha; Sigma) the Student-t limit is used. The relative
# gap between the two densities is O(q) near zero, so 1e-12 keeps the switch
# well under float64 noise while staying far from the Bessel form's small-argument
# regime.
_Q_FLOOR = 1e-12


def _chol_terms(y, mu, Sigma, alpha):
    """Return d^2(y; mu, Sigma), q(alpha; Sigma), (y-mu)'Sigma^-1 alpha and log|Sigma|."""
    L = jnp.linalg.cholesky(Sigma)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    r = y - mu
    sr = jax.scipy.linalg.solve_triangular(L, r[..., None], lower=True)[..., 0]
    sa = jax.scipy.linalg.solve_triangular(L, alpha[..., None], lower=True)[..., 0]
    d2 = jnp.sum(sr * sr, axis=-1)
    q = jnp.sum(sa * sa, axis=-1)
    cross = jnp.sum(sr * sa, axis=-1)
    return d2, q, cross, log_det


def student_t_logpdf(y, mu, Sigma, nu):
    """log density of the multivariate Student-t, the alpha -> 0 limit of Section 2.4."""
    p = jnp.shape(y)[-1]
    d2, _, _, log_det = _chol_terms(y, mu, Sigma, jnp.zeros_like(mu))
    return (
        gammaln(0.5 * (nu + p))
        - gammaln(0.5 * nu)
        - 0.5 * p * jnp.log(nu * jnp.pi)
        - 0.5 * log_det
        - 0.5 * (nu + p) * jnp.log1p(d2 / nu)
    )


def ghst_logpdf(y, mu, Sigma, alpha, nu):
    """log g_GHST,p(y; mu, Sigma, alpha, nu), the density displayed in Section 2.4."""
    y, mu, alpha = (jnp.asarray(v, jnp.float64) for v in (y, mu, alpha))
    Sigma = jnp.asarray(Sigma, jnp.float64)
    nu = jnp.asarray(nu, jnp.float64)
    p = jnp.shape(y)[-1]

    d2, q, cross, log_det = _chol_terms(y, mu, Sigma, alpha)
    # Clamp only the value fed to the Bessel branch; where() alone would still
    # differentiate through a q = 0 argument and produce a nan gradient.
    q_safe = jnp.maximum(q, _Q_FLOOR)
    arg = jnp.sqrt(q_safe * (nu + d2))

    skew = (
        0.5 * nu * jnp.log(nu)
        - 0.5 * p * jnp.log(2.0 * jnp.pi)
        - 0.5 * log_det
        - gammaln(0.5 * nu)
        - (0.5 * nu - 1.0) * jnp.log(2.0)
        + cross
        - 0.25 * (nu + p) * jnp.log((nu + d2) / q_safe)
        + log_bessel_k(0.5 * (nu + p), arg)
    )
    return jnp.where(q > _Q_FLOOR, skew, student_t_logpdf(y, mu, Sigma, nu))


def ghst_mean(mu, alpha, nu):
    """E(X) = mu + E(W) alpha with E(W) = nu/(nu-2), finite for nu > 2 (Section 2.1)."""
    return mu + (nu / (nu - 2.0)) * alpha


def ghst_cov(Sigma, alpha, nu):
    """Cov(X) = E(W) Sigma + Var(W) alpha alpha', the display in Section 2.1.

    Var(W) = 2 nu^2 / ((nu-2)^2 (nu-4)) for W ~ IG(nu/2, nu/2), so this needs
    nu > 4 when alpha is nonzero.
    """
    e_w = nu / (nu - 2.0)
    var_w = 2.0 * nu**2 / ((nu - 2.0) ** 2 * (nu - 4.0))
    return e_w * Sigma + var_w * jnp.outer(alpha, alpha)


def sample_ghst(key, mu, Sigma, alpha, nu, n):
    """Draw n points using the normal mean-variance representation (1)-(2) directly."""
    p = jnp.shape(mu)[-1]
    kw, kz = jax.random.split(key)
    w = sample_inverse_gamma(kw, 0.5 * nu, 0.5 * nu, shape=(n,))
    L = jnp.linalg.cholesky(Sigma)
    z = jax.random.normal(kz, (n, p), dtype=jnp.float64)
    return mu + w[:, None] * alpha + jnp.sqrt(w)[:, None] * (z @ L.T), w


def posterior_w_params(y, mu, Sigma, alpha, nu):
    """(psi, chi, lam) of Proposition 2: (A_s, nu_s + D_js, -(nu_s + p)/2).

    D_js is equation (14) and A_s is equation (15).
    """
    p = jnp.shape(y)[-1]
    d2, q, _, _ = _chol_terms(y, mu, Sigma, alpha)
    return q, nu + d2, -0.5 * (nu + p)
