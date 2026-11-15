"""OU kernel, joint Gram, and conditional GP prior (Section 2.1).

The PDF states the Ornstein-Uhlenbeck kernel in equation (1). It names
squared-exponential (RBF) and periodic kernels, but it does not write a
periodic formula, so this module implements the OU kernel only.
"""

import jax
import jax.numpy as jnp

from mercerflow.common import default_float_dtype


def ou_kernel(s, s_prime, length_scale):
    """Ornstein-Uhlenbeck kernel, equation (1).

    k(s, s') = exp(-|s - s'| / ell).
    """
    return jnp.exp(-jnp.abs(s - s_prime) / length_scale)


def gram_matrix(length, length_scale, jitter, dtype=None):
    """Joint Gram on the integer grid {0, ..., L-1}.

    K_ij = k(i, j) + eps * 1[i = j], with k the OU kernel of equation (1).
    """
    length = int(length)
    if length < 1:
        raise ValueError("length must be positive")
    if float(length_scale) <= 0.0:
        raise ValueError("length_scale must be positive")
    if float(jitter) < 0.0:
        raise ValueError("jitter must be non-negative")
    dtype = default_float_dtype() if dtype is None else dtype
    index = jnp.arange(length, dtype=dtype)
    ell = jnp.asarray(length_scale, dtype=dtype)
    eps = jnp.asarray(jitter, dtype=dtype)
    kernel = ou_kernel(index[:, None], index[None, :], ell)
    return kernel + eps * jnp.eye(length, dtype=dtype)


def partition_gram(K, n_past):
    """Split the joint Gram into past and future blocks, equation (2).

    Returns Kpp, Kpf, Kfp, Kff.
    """
    n_past = int(n_past)
    if not (0 < n_past < K.shape[0]):
        raise ValueError("n_past must leave a nonempty past and future")
    Kpp = K[:n_past, :n_past]
    Kpf = K[:n_past, n_past:]
    Kfp = K[n_past:, :n_past]
    Kff = K[n_past:, n_past:]
    return Kpp, Kpf, Kfp, Kff


def conditional_moments(K, y_past, n_past):
    """Conditional future mean and covariance, equation (3).

    mu = Kfp @ inv(Kpp) @ yp
    Sigma = Kff - Kfp @ inv(Kpp) @ Kpf

    The inverse is applied with a symmetric linear solve, not an explicit
    inverse. y_past has shape (Lp,) or (batch, Lp). Sigma is shared across
    the batch. The returned covariance is symmetrised, matching the
    symmetric Schur complement of a symmetric Gram matrix.
    """
    y_past = jnp.asarray(y_past)
    if y_past.shape[-1] != int(n_past):
        raise ValueError("y_past length must equal n_past")
    Kpp, Kpf, Kfp, Kff = partition_gram(K, n_past)
    # Kpp is symmetric. solve(Kpp, rhs) applies inv(Kpp) without forming it.
    if y_past.ndim == 1:
        alpha = jnp.linalg.solve(Kpp, y_past)
        mu = Kfp @ alpha
    elif y_past.ndim == 2:
        alpha = jnp.linalg.solve(Kpp, y_past.T)
        mu = (Kfp @ alpha).T
    else:
        raise ValueError("y_past must have shape (Lp,) or (batch, Lp)")
    solved_cross = jnp.linalg.solve(Kpp, Kpf)
    sigma = Kff - Kfp @ solved_cross
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma


def _sample_gaussian(key, mean, cov):
    """Draw N(mean, cov) with a Cholesky factor.

    A tiny diagonal jitter is added only so the factor exists when the
    Schur complement has roundoff below zero. conditional_moments itself
    returns equation (3) without this jitter.
    """
    mean = jnp.asarray(mean)
    cov = 0.5 * (cov + cov.T)
    scale = jnp.maximum(jnp.max(jnp.abs(jnp.diag(cov))), jnp.asarray(1.0, dtype=cov.dtype))
    jitter = (jnp.asarray(1e-10, dtype=cov.dtype) * scale) * jnp.eye(
        cov.shape[0], dtype=cov.dtype
    )
    chol = jnp.linalg.cholesky(cov + jitter)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    if mean.ndim == 1:
        return mean + chol @ noise
    return mean + noise @ chol.T


def sample_conditional_future(key, K, y_past, n_past):
    """Sample y0 ~ N(mu, Sigma) from the conditional prior, equation (3)."""
    mu, sigma = conditional_moments(K, y_past, n_past)
    return _sample_gaussian(key, mu, sigma)


def sample_prior(key, K, n):
    """Draw n centred joint windows, y ~ N(0, K), shape (n, L)."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    cov = 0.5 * (K + K.T)
    mean = jnp.zeros((n, K.shape[0]), dtype=K.dtype)
    return _sample_gaussian(key, mean, cov)
