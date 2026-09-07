"""Small working models used in the tests and the demo.

The Gaussian precision model is the textbook case where Hyvarinen's estimator
has a closed form. The logistic location model is the one-dimensional
misspecified example in Section 4 of arXiv:2609.01783. The preferential
sampling model is a scaled-down stand-in for Section 3.1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "vech_indices",
    "vech_to_sym",
    "sym_to_vech",
    "gaussian_precision_log_q",
    "gaussian_precision_closed_form",
    "logistic_log_density",
    "logistic_location_log_q",
    "sample_logistic_location_data",
    "quartic_log_q",
    "prefsamp_log_q",
    "sample_prefsamp",
]


# --- multivariate Gaussian with unknown precision --------------------------


def vech_indices(m: int):
    """Row and column indices of the lower triangle, including the diagonal."""
    return jnp.tril_indices(m)


def vech_to_sym(theta: jnp.ndarray, m: int) -> jnp.ndarray:
    """Unpack a lower-triangle vector into a symmetric m by m matrix."""
    rows, cols = vech_indices(m)
    lower = jnp.zeros((m, m)).at[rows, cols].set(theta)
    return lower + lower.T - jnp.diag(jnp.diag(lower))


def sym_to_vech(mat: jnp.ndarray) -> jnp.ndarray:
    """Pack the lower triangle of a symmetric matrix into a vector."""
    rows, cols = vech_indices(mat.shape[0])
    return mat[rows, cols]


def gaussian_precision_log_q(m: int):
    """log q(x; theta) = -0.5 x' Lambda x for a mean-zero Gaussian.

    The normalizing constant is Z(Lambda) = (2 pi)^{m/2} |Lambda|^{-1/2}, which
    score matching never needs. theta holds the lower triangle of Lambda.
    """

    def log_q(x, theta):
        lam = vech_to_sym(theta, m)
        return -0.5 * x @ lam @ x

    return log_q


def gaussian_precision_closed_form(x: jnp.ndarray) -> jnp.ndarray:
    """The score matching estimator of Lambda in closed form.

    Jhat(Lambda) = -tr(Lambda) + 0.5 tr(Lambda^2 S) with S the sample second
    moment matrix, so the first-order condition is Lambda S + S Lambda = 2 I
    and the minimizer is Lambda = S^{-1}.
    """
    s = x.T @ x / x.shape[0]
    return jnp.linalg.inv(s)


# --- logistic location model, Section 4 ------------------------------------


def logistic_log_density(u: jnp.ndarray) -> jnp.ndarray:
    """log f(u) with f(u) = e^{-u} / (1 + e^{-u})^2, the standard logistic."""
    return -u - 2.0 * jax.nn.softplus(-u)


def logistic_location_log_q(x, theta):
    """The working model p(x; theta) = f(x - theta) of Section 4.

    x has shape (1,) and theta has shape (1,).
    """
    return logistic_log_density(x[0] - theta[0])


def sample_logistic_location_data(key, n: int, loc: float = 1.0, scale: float = 4.0):
    """Draw from p_x(x) = 0.25 f(0.25 (x - 1)), the Section 4 truth.

    That is x = 1 + 4 L with L standard logistic. The working model f(x - theta)
    is misspecified because the data are four times as dispersed. Both the KL
    target and the score matching target sit at theta = 1 by symmetry.
    """
    u = jax.random.uniform(key, (n, 1), minval=1e-12, maxval=1 - 1e-12)
    return loc + scale * jnp.log(u / (1.0 - u))


# --- a nonlinear unnormalized model ----------------------------------------


def quartic_log_q(x, theta):
    """log q(x; theta) = -theta0 ||x||^2 / 2 - theta1 sum x_j^4 - theta2 x_0 x_1.

    A genuinely unnormalized model: Z(theta) has no closed form. Used to check
    the autodiff moment function against finite differences.
    """
    return (
        -0.5 * theta[0] * jnp.sum(x**2)
        - theta[1] * jnp.sum(x**4)
        - theta[2] * x[0] * x[1]
    )


# --- scaled-down preferential sampling model, Section 3.1 ------------------


def _xi(s, beta):
    return beta[0] * s[0] + beta[1] * s[1]


def prefsamp_log_q(y, s, theta):
    """Joint log density of a response-location pair, up to a theta-only factor.

    theta = (beta1, beta2, a, eta0, tau) with tau = log sigma^2. The location
    density is p(s) proportional to exp{xi(s; beta)} on [0, 1]^2 and the
    response is y | s ~ N(eta0 + a xi(s), sigma^2). This is the same structure
    as eq (3.1), with the low-rank Gaussian process surfaces replaced by a
    linear trend so the whole thing has five parameters instead of 76.

    The spatial normalizing integral is dropped, which is legitimate here
    because eq (2.9) only ever differentiates in (y, s).
    """
    beta = theta[0:2]
    a, eta0, tau = theta[2], theta[3], theta[4]
    xi = _xi(s, beta)
    mu = eta0 + a * xi
    return -0.5 * (y[0] - mu) ** 2 * jnp.exp(-tau) + xi


def _sample_truncexp(key, n, rate):
    """Draw from a density proportional to exp(rate * t) on [0, 1]."""
    u = jax.random.uniform(key, (n,))
    linear = u
    tilted = jnp.log1p(u * jnp.expm1(rate)) / jnp.where(rate == 0.0, 1.0, rate)
    return jnp.where(jnp.abs(rate) < 1e-8, linear, tilted)


def sample_prefsamp(key, n, theta, df: float = 5.0, misspecified: bool = True):
    """Simulate from the preferential sampling truth.

    Locations come from the working model's own p(s). The response is Gaussian
    when misspecified is False, and otherwise mu(s) plus a central t scaled to
    unit variance, so the working model's Gaussian tails are wrong. Section 3.1
    uses a noncentral t_5 instead, which also shifts the mean away from mu(s).
    """
    k_s1, k_s2, k_y = jax.random.split(key, 3)
    beta = theta[0:2]
    a, eta0, tau = theta[2], theta[3], theta[4]
    sigma = jnp.exp(0.5 * tau)
    s = jnp.stack(
        [_sample_truncexp(k_s1, n, beta[0]), _sample_truncexp(k_s2, n, beta[1])], axis=1
    )
    xi = jax.vmap(lambda si: _xi(si, beta))(s)
    mu = eta0 + a * xi
    if misspecified:
        raw = jax.random.t(k_y, df, (n,))
        noise = raw / jnp.sqrt(df / (df - 2.0))
    else:
        noise = jax.random.normal(k_y, (n,))
    y = (mu + sigma * noise)[:, None]
    return y, s
