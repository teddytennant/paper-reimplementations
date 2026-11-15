"""L2-regularized logistic propensity with Laplace output perturbation (§4.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _sigmoid(z):
    return jax.nn.sigmoid(z)


def _add_intercept(x):
    x = jnp.asarray(x)
    ones = jnp.ones((x.shape[0], 1), dtype=x.dtype)
    return jnp.concatenate([ones, x], axis=1)


def fit_logistic_l2(x, w, lam=1.0, n_iter=40):
    """Newton / IRLS for mean BCE + (λ/2) ||θ||^2, including intercept.

    λ=1 matches the paper's regularization constant of 1 together with
    output-perturbation scale 2/(n ε) (Chaudhuri et al. 2009).
    """
    x = jnp.asarray(x)
    w = jnp.asarray(w, dtype=x.dtype)
    z = _add_intercept(x)
    n, p = z.shape
    lam = jnp.asarray(lam, dtype=x.dtype)
    theta = jnp.zeros((p,), dtype=x.dtype)

    def body(_, theta):
        logits = z @ theta
        p_hat = _sigmoid(logits)
        p_hat = jnp.clip(p_hat, 1e-12, 1.0 - 1e-12)
        grad = (z.T @ (p_hat - w)) / n + lam * theta
        weights = p_hat * (1.0 - p_hat)
        hess = (z.T * weights) @ z / n + lam * jnp.eye(p, dtype=x.dtype)
        step = jnp.linalg.solve(hess, grad)
        return theta - step

    theta = jax.lax.fori_loop(0, int(n_iter), body, theta)
    return theta


def predict_propensity(theta, x, eps_num=1e-15):
    """Predicted ê(X) clipped to (eps_num, 1 - eps_num) so scores lie in (0, 1)."""
    x = jnp.asarray(x)
    theta = jnp.asarray(theta, dtype=x.dtype)
    z = _add_intercept(x)
    p = _sigmoid(z @ theta)
    return jnp.clip(p, eps_num, 1.0 - eps_num)


def _sample_laplace(key, shape, scale):
    scale = jnp.asarray(scale)
    u = jax.random.uniform(
        key, shape, minval=-0.5, maxval=0.5, dtype=jnp.result_type(scale, jnp.float64)
    )
    u = jnp.clip(u, -0.5 + 1e-15, 0.5 - 1e-15)
    return -scale * jnp.sign(u) * jnp.log1p(-2.0 * jnp.abs(u))


def fit_propensity(x, w, key=None, eps=None, add_noise=True, lam=1.0, n_iter=40):
    """Fit L2 logistic regression, optionally adding Lap(2/(n ε)) to each coefficient.

    Pass add_noise=False, eps=None, or non-finite eps for a noise-free fit.
    """
    x = jnp.asarray(x)
    w = jnp.asarray(w)
    theta = fit_logistic_l2(x, w, lam=lam, n_iter=n_iter)
    n = x.shape[0]
    if (not add_noise) or (key is None) or (eps is None):
        return theta
    scale = 2.0 / (n * jnp.asarray(eps, dtype=theta.dtype))
    scale = jnp.where(jnp.isfinite(scale), scale, 0.0)
    return theta + _sample_laplace(key, theta.shape, scale)
