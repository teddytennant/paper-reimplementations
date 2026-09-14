"""Unnormalized log-ratio z, empirical log-ratio variation X, and KLXX (eqs. 5-7, Alg. 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def log_ratio_z(u0_of_gy, u_of_y, logdet_g):
    """Unnormalized log-ratio z(y) = U0(G(y)) - U(y) - log|det JG(y)|.

    Additive constants independent of y cancel in X and shift the forward-KL
    surrogate by a fixed amount, so they may be omitted from U0 and U.
    """
    return u0_of_gy - u_of_y - logdet_g


def flow_log_ratio(y, g_y, logdet_g, u_of_y, u0_of_gy):
    """z on a batch or a single point, from already-evaluated potentials and logdet."""
    return log_ratio_z(u0_of_gy, u_of_y, logdet_g)


def log_ratio_variation(z):
    """Empirical U-statistic X-hat = 1/(n(n-1)) sum_{i != j} |z_i - z_j|.

    The diagonal is zero and is excluded from the denominator. n >= 2.
    """
    n = z.shape[0]
    pairwise = jnp.abs(z[:, None] - z[None, :])
    return pairwise.sum() / (n * (n - 1))


def klxx_loss(z_pi, z_xi, lam=1.0, theta=1.0):
    """Empirical KLXX (Alg. 1 / eq. 19): mean(z) on X_pi plus two variations.

    L-hat[G] = mean_{y in X_pi} z(y) + lam * X-hat(z on X_pi) + theta * X-hat(z on X_xi).

    Default (lam, theta) = (1, 1). Setting both to 0 recovers the MC forward-KL
    surrogate mean(z) on the pi-batch.
    """
    forward_kl = jnp.mean(z_pi)
    return forward_kl + lam * log_ratio_variation(z_pi) + theta * log_ratio_variation(z_xi)


def z_from_flow(params, y, U, U0, forward_fn):
    """z(y) for a coupling flow: G, logdet from forward_fn, potentials U0 and U."""
    gy, logdet = forward_fn(params, y)

    def _potentials(yy, gyy):
        return U0(gyy), U(yy)

    if y.ndim == 1:
        u0, u = _potentials(y, gy)
        return log_ratio_z(u0, u, logdet)
    u0, u = jax.vmap(_potentials)(y, gy)
    return log_ratio_z(u0, u, logdet)
