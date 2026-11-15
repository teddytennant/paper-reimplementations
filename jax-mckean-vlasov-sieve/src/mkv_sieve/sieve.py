"""The ReQU sieve of Section 4.1, eq (4.1).

    xi_theta(r) = kappa + (K - kappa) S_tau(N_theta(r)),   r in [0, U],

with N_theta a ReQU network, sigma(z) = (z_+)^2, and S_tau a globally Lipschitz
clamp onto [0, 1] that is the identity on [tau, 1 - tau].  For r > U the network
is extended by the constant xi_theta(U), which is what makes the bounds
kappa <= xi <= K of Assumption 3.3 hold on all of R+ rather than just on [0, U].

The domain bound is U >= max(1, a_+) with a_+ = Z_K^{-1}, so for the OU
potential and the constants used in the paper's simulation study U = 1.

ReQU rather than ReLU because Section 4.2 differentiates the forward map through
xi, so the sieve elements have to be C^1.  It also makes quadratics exact:
z^2 = sigma(z) + sigma(-z) and, for |z| < c, z = (sigma(z + c) - sigma(c - z)) / (4c).
Those two identities are the "two ReQU units for a linear term, one per knot"
count in the paper's Section 8 parametrisation.
"""

import math
from typing import Callable, Sequence

import jax
import jax.numpy as jnp


def requ(z):
    """ReQU activation sigma(z) = (z_+)^2."""
    zp = jnp.maximum(z, 0.0)
    return zp * zp


def clamp_tau(z, tau: float):
    """S_tau: R -> (0, 1), 1-Lipschitz, C^1, and the identity on [tau, 1 - tau].

    Outside that window it decays exponentially onto the endpoints, matching
    value and slope at tau and 1 - tau.  A hard clip would also satisfy the
    conditions in eq (4.1) but has zero gradient once a unit saturates, which
    strands the optimiser; this version never does.
    """
    lo = tau * jnp.exp(jnp.minimum((z - tau) / tau, 0.0))
    hi = 1.0 - tau * jnp.exp(jnp.minimum((1.0 - tau - z) / tau, 0.0))
    return jnp.where(z < tau, lo, jnp.where(z > 1.0 - tau, hi, z))


def requ_net(params: Sequence, r):
    """N_theta(r), a fully connected ReQU network, eq (4.1).

    params is a list of (W, b) with W of shape (out, in).  ReQU on every hidden
    layer, identity on the output.  Input is a scalar array, broadcast over any
    shape.
    """
    h = jnp.expand_dims(jnp.asarray(r), -1)
    for W, b in params[:-1]:
        h = requ(h @ W.T + b)
    W, b = params[-1]
    return jnp.squeeze(h @ W.T + b, -1)


def init_requ_params(key, width: int = 6, U: float = 1.0, depth: int = 2):
    """Quadratic-hinge initialisation for a depth-`depth` ReQU network.

    The hidden units start as knots (t_j - r)_+^2 spread over [0, U], which is
    the family the paper's simulation study uses, and the output layer starts
    near 0.5 so xi_theta starts in the middle of [kappa, K] where S_tau is the
    identity.
    """
    keys = jax.random.split(key, depth + 1)
    knots = jnp.linspace(0.15 * U, U, width)
    W1 = -jnp.ones((width, 1))
    b1 = knots
    params = [(W1, b1)]
    w = width
    for i in range(depth - 2):
        Wi = 0.3 * jax.random.normal(keys[i], (width, w)) / jnp.sqrt(w)
        params.append((Wi, jnp.zeros(width)))
        w = width
    Wl = 0.05 * jax.random.normal(keys[depth - 1], (1, w)) / jnp.sqrt(w)
    params.append((Wl, jnp.full((1,), 0.5)))
    return params


def make_xi(params, kappa: float, K: float, tau: float = 0.05, U: float = 1.0
            ) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build xi_theta of eq (4.1), extended by xi_theta(U) above U."""

    def xi(r):
        rc = jnp.minimum(jnp.asarray(r), U)
        return kappa + (K - kappa) * clamp_tau(requ_net(params, rc), tau)

    return xi


def domain_bound(pot, K: float) -> float:
    """U = max(1, a_+) with a_+ = Z_K^{-1}, Section 4.1."""
    return max(1.0, math.exp(-pot.log_Z(K)))
