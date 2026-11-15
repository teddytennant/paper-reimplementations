"""Matrix normal density and sampling.

The convention is the one in Section 1 of the paper: for a p x q matrix Y,

    phi(Y | M, S1, S2, sigma^2)
        = exp{-tr(Xi_1 (Y - M) Xi_2 (Y - M)') / (2 sigma^2)}
          / [(2 pi sigma^2)^{pq/2} |S1|^{q/2} |S2|^{p/2}],

which is the density displayed in Section 4.1, and equivalently, by (1),

    vec(Y) ~ N_pq(vec(M), sigma^2 (S2 (x) S1)).

vec stacks columns, so entry (i, j) of Y sits at index j * p + i of vec(Y) and
the Kronecker factor order is S2 (x) S1, column factor first. Getting that order
backwards is the classic way to be wrong here, so `matnorm_logpdf` is checked
against `mvn_logpdf` on the explicit Kronecker product in the test suite.
"""

import jax.numpy as jnp
import jax.random as jr


def vec(Y):
    """Column-stacking vectorization: vec(Y)[j * p + i] = Y[i, j]."""
    return jnp.asarray(Y).T.reshape(-1)


def unvec(y, p, q):
    """Inverse of `vec` for a p x q matrix."""
    return jnp.asarray(y).reshape(q, p).T


def kron_cov(S1, S2, sigma2):
    """The full pq x pq covariance sigma^2 (S2 (x) S1) of (1)."""
    return sigma2 * jnp.kron(S2, S1)


def matnorm_logpdf(Y, params):
    """Log of the matrix normal density, in matrix form.

    Uses Cholesky factors of the p x p and q q x q factors only; the pq x pq
    covariance is never built. With |S1| = |S2| = 1 the log determinant term
    collapses to pq log sigma^2, which is the identity used in Section 4.1.
    """
    M, S1, S2, sigma2 = params
    p, q = M.shape
    D = jnp.asarray(Y) - M
    L1 = jnp.linalg.cholesky(S1)
    L2 = jnp.linalg.cholesky(S2)
    # tr(Xi_1 D Xi_2 D') = ||L1^{-1} D L2^{-T}||_F^2
    A = jnp.linalg.solve(L1, D)
    A = jnp.linalg.solve(L2, A.T).T
    quad = jnp.sum(A * A) / sigma2
    logdet1 = 2.0 * jnp.sum(jnp.log(jnp.diag(L1)))
    logdet2 = 2.0 * jnp.sum(jnp.log(jnp.diag(L2)))
    return -0.5 * (
        p * q * jnp.log(2.0 * jnp.pi * sigma2)
        + q * logdet1
        + p * logdet2
        + quad
    )


def mvn_logpdf(y, mu, Sigma):
    """Plain multivariate normal log density, for checking (1) directly."""
    y = jnp.asarray(y)
    d = y.shape[-1]
    L = jnp.linalg.cholesky(Sigma)
    z = jnp.linalg.solve(L, y - mu)
    return -0.5 * (
        d * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        + jnp.sum(z * z)
    )


def sample_matnorm(key, params, n):
    """Draw n matrices from N_{p x q}(M, sigma^2 S1, S2).

    Y = M + sqrt(sigma^2) L1 E L2', with E standard normal, gives
    vec(Y) = vec(M) + sqrt(sigma^2) (L2 (x) L1) vec(E), which has covariance
    sigma^2 (S2 (x) S1) as required by (1).
    """
    M, S1, S2, sigma2 = params
    p, q = M.shape
    L1 = jnp.linalg.cholesky(S1)
    L2 = jnp.linalg.cholesky(S2)
    E = jr.normal(key, (n, p, q), dtype=jnp.float64)
    return M + jnp.sqrt(sigma2) * jnp.einsum("ab,nbc,dc->nad", L1, E, L2)
