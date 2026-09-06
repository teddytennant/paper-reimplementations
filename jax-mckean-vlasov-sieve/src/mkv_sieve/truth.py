"""The three test coefficients of the paper's simulation study, Section 8.

    Xi_0,1(r) = 1
    Xi_0,2(r) = 0.8 + 0.4 (0.5 - 0.1 (1 - r)_+^2)
    Xi_0,3(r) = 0.8 + 0.4 r / (1 + r)

with kappa = 0.8, K = 1.2 for the first two and kappa = 0.7, K = 1.3 for the
third.  The first two sit inside the smallest ReQU family the paper considers;
the third does not, so it also carries an approximation error.
"""

import jax.numpy as jnp

ENVELOPE = {1: (0.8, 1.2), 2: (0.8, 1.2), 3: (0.7, 1.3)}


def xi_1(r):
    return jnp.full_like(jnp.asarray(r, dtype=float), 1.0)


def xi_2(r):
    return 0.8 + 0.4 * (0.5 - 0.1 * jnp.maximum(1.0 - jnp.asarray(r), 0.0) ** 2)


def xi_3(r):
    r = jnp.asarray(r)
    return 0.8 + 0.4 * r / (1.0 + r)


TRUTH = {1: xi_1, 2: xi_2, 3: xi_3}
