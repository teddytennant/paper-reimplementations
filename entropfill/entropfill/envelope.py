"""Mixture-to-row deletion envelope, Theorem 4.1.

For a proposal row a with mixture weight omega and pooled mass q on a
removal set, the row mass eta on that set is at most
min(1, q/omega, q + sqrt(D q (1-q))), and D_chi2(a || mu) is at most
1/omega - 1.
"""

import jax.numpy as jnp


def chi2(row, pooled):
    """D_chi2(a || mu) on the support of mu, equation (11)."""
    support = pooled > 0
    ratio = jnp.where(support, row**2 / jnp.maximum(pooled, 1e-30), 0.0)
    return jnp.sum(ratio) - 1.0


def deletion_envelope(q, omega, divergence):
    """The three-term bound of Theorem 4.1."""
    third = q + jnp.sqrt(jnp.maximum(divergence, 0.0) * q * (1.0 - q))
    return jnp.minimum(1.0, jnp.minimum(q / jnp.maximum(omega, 1e-30), third))


def envelope_holds(row, pooled, omega, removal, tol=1e-5):
    """True when the row mass on `removal` respects the envelope."""
    q = jnp.sum(pooled * removal)
    eta = jnp.sum(row * removal)
    bound = deletion_envelope(q, omega, chi2(row, pooled))
    divergence_bound = 1.0 / jnp.maximum(omega, 1e-30) - 1.0
    return (eta <= bound + tol) & (chi2(row, pooled) <= divergence_bound + tol)
