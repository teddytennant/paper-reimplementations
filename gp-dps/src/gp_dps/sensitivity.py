"""Derivative-process posterior (eqs 2--3) and sensitivity estimator (eq 4)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve

from gp_dps.gp import alpha_from_chol, cho_kn
from gp_dps.kernel import prior_deriv_variance, se_ard_deriv_first


def deriv_kernel_train(X: Array, ell: Array, eta: Array, j: int | Array) -> Array:
    """(∂j K⋆)_{i,s} = ∂k(x_i, x_s) / ∂(x_i)_j  (first-argument derivative)."""
    return se_ard_deriv_first(X, X, ell, eta, j)


def all_deriv_kernel_train(X: Array, ell: Array, eta: Array) -> Array:
    """Stack of ∂j K⋆ over j. Shape (D, n, n)."""
    d = X.shape[1]
    return jax.vmap(lambda j: deriv_kernel_train(X, ell, eta, j))(jnp.arange(d))


def A_maps(dK: Array, L: Array) -> Array:
    """A_j = (∂j K⋆) Kn^{-1} for each j. dK: (D, n, n) -> A: (D, n, n)."""

    def one(dKj):
        return cho_solve((L, True), dKj.T).T

    return jax.vmap(one)(dK)


def deriv_mean_at(X: Array, alpha: Array, x: Array, ell: Array, eta: Array, j: int | Array) -> Array:
    """Eq 2: ∂j f(x) = (∂j k_x)^T alpha, with (∂j k_x)_i = ∂k(x, x_i)/∂x_j."""
    x = jnp.atleast_2d(x)
    djk = se_ard_deriv_first(x, X, ell, eta, j)
    return (djk @ alpha).reshape(-1)


def deriv_var_at(X: Array, L: Array, x: Array, ell: Array, eta: Array, j: int | Array) -> Array:
    """Eq 3: Var(∂f/∂x_j (x) | data) = eta/ell_j^2 - (∂j k_x)^T Kn^{-1} (∂j k_x)."""
    x = jnp.atleast_2d(x)
    djk = se_ard_deriv_first(x, X, ell, eta, j)
    prior = prior_deriv_variance(ell, eta)[j]
    w = cho_solve((L, True), djk.T)
    quad = jnp.sum(djk * w.T, axis=-1)
    return prior - quad


def deriv_mean_train(dK: Array, alpha: Array) -> Array:
    """Eq 2 at the training inputs: (n,) per j, stacked as (D, n)."""
    return dK @ alpha


def deriv_var_train(dK: Array, A: Array, ell: Array, eta: Array) -> Array:
    """Eq 3 at the training inputs. dK, A: (D, n, n). Returns (D, n)."""
    prior = prior_deriv_variance(ell, eta)[:, None]
    quad = jnp.sum(A * dK, axis=-1)
    return prior - quad


def plugin_quadratic(A: Array, y: Array) -> Array:
    """Plug-in term (1/n) ||A_j y||^2 = y^T A_j^T A_j y / n. Returns (D,)."""
    gy = A @ y
    n = y.shape[0]
    return jnp.sum(gy ** 2, axis=-1) / n


def sensitivity_nu(X: Array, y: Array, ell: Array, eta: Array, sigma2: Array) -> Array:
    """Eq 4: nu_j = (1/n) sum_i [∂j fbar(x_i)]^2 + (1/n) sum_i Var(∂f/∂x_j(x_i)|data)."""
    L = cho_kn(X, ell, eta, sigma2)
    alpha = alpha_from_chol(L, y)
    dK = all_deriv_kernel_train(X, ell, eta)
    A = A_maps(dK, L)
    plugin = plugin_quadratic(A, y)
    v = deriv_var_train(dK, A, ell, eta)
    return plugin + jnp.mean(v, axis=-1)


def dps_maps(X: Array, y: Array, ell: Array, eta: Array, sigma2: Array) -> dict:
    """Shared maps for Algorithm 1: Kn chol, alpha, A_j, nu, training derivatives."""
    L = cho_kn(X, ell, eta, sigma2)
    alpha = alpha_from_chol(L, y)
    dK = all_deriv_kernel_train(X, ell, eta)
    A = A_maps(dK, L)
    gbar = deriv_mean_train(dK, alpha)
    v = deriv_var_train(dK, A, ell, eta)
    n = y.shape[0]
    plugin = jnp.sum(gbar ** 2, axis=-1) / n
    nu = plugin + jnp.mean(v, axis=-1)
    return {
        "L": L,
        "alpha": alpha,
        "dK": dK,
        "A": A,
        "gbar": gbar,
        "v": v,
        "plugin": plugin,
        "nu": nu,
    }
