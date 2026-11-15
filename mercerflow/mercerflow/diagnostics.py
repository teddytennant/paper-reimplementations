"""Proposition 2 Jacobi condition number, no training required.

For a centred prior y ~ N(0, K) and x = Phi.T @ y,
A = E[x x^T] = Phi.T @ K @ Phi.
The Jacobi preconditioner is D = diag(A)^{-1/2}.
kappa(D A D) is 1 for the Mercer codec, up to numerical jitter, because
A is the diagonal eigenvalue matrix.
"""

import jax.numpy as jnp


def prior_coefficient_covariance(Phi, K):
    """A = E[x x^T] for x = Phi.T @ y, y ~ N(0, K)."""
    return Phi.T @ K @ Phi


def jacobi_scaled(A):
    """G = D A D with D = diag(A)^{-1/2}."""
    diag = jnp.diag(A)
    scale = 1.0 / jnp.sqrt(diag)
    scaled = (scale[:, None] * A) * scale[None, :]
    return 0.5 * (scaled + scaled.T)


def jacobi_condition_number(A):
    """kappa(D A D) = lambda_max / lambda_min of the Jacobi-scaled matrix."""
    scaled = jacobi_scaled(A)
    eigenvalues = jnp.linalg.eigvalsh(scaled)
    return eigenvalues[-1] / eigenvalues[0]
