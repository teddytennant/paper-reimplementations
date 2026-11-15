"""Exact ARD squared-exponential Gaussian process and analytic EI."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


_JITTER = 1e-6


def standardize(y):
    """Return (y_std, mean, std) with std floored at 1e-12."""
    y = jnp.asarray(y)
    mean = jnp.mean(y)
    std = jnp.maximum(jnp.std(y), 1e-12)
    return (y - mean) / std, mean, std


def _kernel(x1, x2, lengthscales, signal_var):
    # x1: (n, d), x2: (m, d) -> (n, m)
    scaled = (x1[:, None, :] - x2[None, :, :]) / lengthscales
    return signal_var * jnp.exp(-0.5 * jnp.sum(scaled * scaled, axis=-1))


def _train_cov(x_train, lengthscales, signal_var, noise_var):
    n = x_train.shape[0]
    cov = _kernel(x_train, x_train, lengthscales, signal_var)
    return cov + (noise_var + _JITTER) * jnp.eye(n, dtype=cov.dtype)


def posterior(x_train, y_train, x, lengthscales, signal_var, noise_var):
    """Posterior mean and variance of the latent function at x.

    var is the latent variance (observation noise is only on the training
    diagonal) and is clipped to be at least 1e-12. x may be (d,) or (n, d).
    """
    x_train = jnp.atleast_2d(jnp.asarray(x_train))
    y_train = jnp.asarray(y_train).reshape(-1)
    lengthscales = jnp.asarray(lengthscales)
    signal_var = jnp.asarray(signal_var)
    noise_var = jnp.asarray(noise_var)
    x_in = jnp.asarray(x)
    single = x_in.ndim == 1
    x_q = jnp.atleast_2d(x_in)

    chol = jnp.linalg.cholesky(_train_cov(x_train, lengthscales, signal_var, noise_var))
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    cross = _kernel(x_train, x_q, lengthscales, signal_var)
    mu = cross.T @ alpha
    solved = jax.scipy.linalg.cho_solve((chol, True), cross)
    var = signal_var - jnp.sum(cross * solved, axis=0)
    var = jnp.maximum(var, 1e-12)
    if single:
        return mu[0], var[0]
    return mu, var


def neg_log_marginal_likelihood(x_train, y_train, lengthscales, signal_var, noise_var):
    """0.5 y^T K^{-1} y + 0.5 log|K| + 0.5 n log(2 pi), K includes noise and jitter."""
    x_train = jnp.atleast_2d(jnp.asarray(x_train))
    y_train = jnp.asarray(y_train).reshape(-1)
    n = y_train.shape[0]
    chol = jnp.linalg.cholesky(_train_cov(x_train, lengthscales, signal_var, noise_var))
    alpha = jax.scipy.linalg.cho_solve((chol, True), y_train)
    data_fit = 0.5 * jnp.dot(y_train, alpha)
    logdet = jnp.sum(jnp.log(jnp.diag(chol)))
    const = 0.5 * n * jnp.log(2.0 * jnp.pi)
    return data_fit + logdet + const


def fit_mll(x_train, y_train, steps=40, lr=0.05, seed=0):
    """Adam on log-lengthscales, log-signal, and log-noise.

    Starts at lengthscales=0.5, signal_var=1, noise_var=0.1. Deterministic
    given seed (the start point is fixed; seed only anchors the PRNG).
    """
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train).reshape(-1)
    d = x_train.shape[-1]
    # Consume the seed so the call is a pure function of it even though the
    # initialization itself does not use randomness.
    _ = jax.random.PRNGKey(seed)

    log_ls = jnp.log(jnp.full((d,), 0.5))
    log_signal = jnp.log(jnp.asarray(1.0))
    log_noise = jnp.log(jnp.asarray(0.1))

    def loss(log_lengthscales, log_signal_var, log_noise_var):
        return neg_log_marginal_likelihood(
            x_train,
            y_train,
            jnp.exp(log_lengthscales),
            jnp.exp(log_signal_var),
            jnp.exp(log_noise_var),
        )

    nll_before = loss(log_ls, log_signal, log_noise)
    grad_loss = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

    m_ls = jnp.zeros_like(log_ls)
    v_ls = jnp.zeros_like(log_ls)
    m_signal = jnp.zeros_like(log_signal)
    v_signal = jnp.zeros_like(log_signal)
    m_noise = jnp.zeros_like(log_noise)
    v_noise = jnp.zeros_like(log_noise)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for t in range(1, int(steps) + 1):
        g_ls, g_signal, g_noise = grad_loss(log_ls, log_signal, log_noise)
        m_ls = beta1 * m_ls + (1.0 - beta1) * g_ls
        v_ls = beta2 * v_ls + (1.0 - beta2) * (g_ls * g_ls)
        m_signal = beta1 * m_signal + (1.0 - beta1) * g_signal
        v_signal = beta2 * v_signal + (1.0 - beta2) * (g_signal * g_signal)
        m_noise = beta1 * m_noise + (1.0 - beta1) * g_noise
        v_noise = beta2 * v_noise + (1.0 - beta2) * (g_noise * g_noise)
        corr1 = 1.0 - beta1 ** t
        corr2 = 1.0 - beta2 ** t
        log_ls = log_ls - lr * (m_ls / corr1) / (jnp.sqrt(v_ls / corr2) + eps)
        log_signal = log_signal - lr * (m_signal / corr1) / (jnp.sqrt(v_signal / corr2) + eps)
        log_noise = log_noise - lr * (m_noise / corr1) / (jnp.sqrt(v_noise / corr2) + eps)

    nll_after = loss(log_ls, log_signal, log_noise)
    return {
        "lengthscales": jnp.exp(log_ls),
        "signal_var": jnp.exp(log_signal),
        "noise_var": jnp.exp(log_noise),
        "nll_before": nll_before,
        "nll_after": nll_after,
    }


def expected_improvement(mu, var, y_best):
    """Analytic expected improvement for minimization.

    EI = (y_best - mu) * Phi(z) + sigma * phi(z), z = (y_best - mu) / sigma.
    If sigma <= 1e-12, EI = max(y_best - mu, 0).
    """
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)
    sigma = jnp.sqrt(jnp.maximum(var, 0.0))
    safe_sigma = jnp.maximum(sigma, 1e-12)
    z = (y_best - mu) / safe_sigma
    ei = (y_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    fallback = jnp.maximum(y_best - mu, 0.0)
    return jnp.where(sigma > 1e-12, ei, fallback)
