"""Zero-mean SE-ARD GP: Kn, alpha, log marginal likelihood, hyperparameter fit."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import softplus
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular

from gp_dps.kernel import se_ard

_MIN_POS = 1e-6


def kernel_matrix(X: Array, ell: Array, eta: Array) -> Array:
    """Noise-free Gram matrix K on the training inputs."""
    return se_ard(X, X, ell, eta)


def kn_matrix(X: Array, ell: Array, eta: Array, sigma2: Array) -> Array:
    """Kn = K + sigma^2 I."""
    n = X.shape[0]
    return kernel_matrix(X, ell, eta) + sigma2 * jnp.eye(n, dtype=X.dtype)


def cho_kn(X: Array, ell: Array, eta: Array, sigma2: Array):
    """Cholesky factor of Kn (lower triangular)."""
    Kn = kn_matrix(X, ell, eta, sigma2)
    L, _ = cho_factor(Kn, lower=True)
    return L


def alpha_from_chol(L: Array, y: Array) -> Array:
    """alpha = Kn^{-1} y."""
    return cho_solve((L, True), y)


def kn_inv_from_chol(L: Array) -> Array:
    """Full Kn^{-1} from its Cholesky factor."""
    n = L.shape[0]
    return cho_solve((L, True), jnp.eye(n, dtype=L.dtype))


def kn_inv_diag(L: Array) -> Array:
    """Diagonal of Kn^{-1}: ||L^{-1} e_i||^2."""
    Linv = solve_triangular(L, jnp.eye(L.shape[0], dtype=L.dtype), lower=True)
    return jnp.sum(Linv ** 2, axis=0)


def log_marginal_likelihood(X: Array, y: Array, ell: Array, eta: Array, sigma2: Array) -> Array:
    """log p(y | X, theta) = -1/2 y^T Kn^{-1} y - 1/2 log|Kn| - n/2 log(2 pi)."""
    L = cho_kn(X, ell, eta, sigma2)
    alpha = alpha_from_chol(L, y)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    n = y.shape[0]
    return -0.5 * jnp.dot(y, alpha) - 0.5 * logdet - 0.5 * n * jnp.log(2.0 * jnp.pi)


def posterior_mean(X: Array, alpha: Array, Xstar: Array, ell: Array, eta: Array) -> Array:
    """fbar(x*) = k_{x*}^T alpha."""
    kx = se_ard(Xstar, X, ell, eta)
    return kx @ alpha


def posterior_cov(X: Array, L: Array, Xstar: Array, ell: Array, eta: Array) -> Array:
    """cov(f(Xstar) | data) = K(Xstar, Xstar) - k_*^T Kn^{-1} k_*."""
    Kss = se_ard(Xstar, Xstar, ell, eta)
    kstar = se_ard(Xstar, X, ell, eta)
    v = cho_solve((L, True), kstar.T)
    return Kss - kstar @ v


def _inv_softplus(z: Array) -> Array:
    z = jnp.maximum(z, _MIN_POS)
    return jnp.log(jnp.expm1(z))


def unpack_raw(raw: Array, d: int) -> tuple[Array, Array, Array]:
    """Map unconstrained raw parameters to (ell, eta, sigma2) via softplus."""
    ell = softplus(raw[:d]) + _MIN_POS
    eta = softplus(raw[d]) + _MIN_POS
    sigma2 = softplus(raw[d + 1]) + _MIN_POS
    return ell, eta, sigma2


def pack_raw(ell: Array, eta: Array, sigma2: Array) -> Array:
    return jnp.concatenate(
        [
            _inv_softplus(ell - _MIN_POS),
            _inv_softplus(jnp.atleast_1d(eta - _MIN_POS)),
            _inv_softplus(jnp.atleast_1d(sigma2 - _MIN_POS)),
        ]
    )


def _nll(raw: Array, X: Array, y: Array) -> Array:
    d = X.shape[1]
    ell, eta, sigma2 = unpack_raw(raw, d)
    return -log_marginal_likelihood(X, y, ell, eta, sigma2)


def fit_se_ard(
    X: Array,
    y: Array,
    steps: int = 80,
    lr: float = 0.08,
    ell0: Array | None = None,
    eta0: Array | None = None,
    sigma20: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Maximize SE-ARD log marginal likelihood by gradient descent on softplus params."""
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    d = X.shape[1]
    yvar = jnp.maximum(jnp.var(y), jnp.array(1e-4, dtype=y.dtype))
    if ell0 is None:
        ell0 = jnp.ones(d, dtype=X.dtype)
    if eta0 is None:
        eta0 = yvar
    if sigma20 is None:
        sigma20 = 0.1 * yvar
    raw = pack_raw(jnp.asarray(ell0, dtype=X.dtype), jnp.asarray(eta0, dtype=X.dtype), jnp.asarray(sigma20, dtype=X.dtype))

    def body(_, raw):
        g = jax.grad(_nll)(raw, X, y)
        g = jnp.clip(g, -20.0, 20.0)
        return raw - lr * g

    raw = jax.lax.fori_loop(0, steps, body, raw)
    return unpack_raw(raw, d)
