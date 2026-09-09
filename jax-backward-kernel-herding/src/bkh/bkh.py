"""Backward Kernel Herding (Algorithms 6 and 7, arXiv:2609.07432).

Sign convention
---------------
Algorithm 6 line 5 selects
    r_m = argmin_r ||Phi(x_r) - mu_n - eps||^2
while equation (14) writes ||Phi(x_r) - mu_n + eps_t||^2. Equation (16) and
Algorithm 7 agree with the minus sign in Algorithm 6, so that implementation
is used here. The residual update is taken from Algorithm 6 line 6 / eq (15):
    eps <- Phi(x_rm) - mu_n + eps
which is the source of truth for the MMD identity in Section 3.2.1:
    eps_m = sum_{x in R_m} Phi(x) - m * mu_n
    MMD^2(X^{(m)}, mu_n) = ||mu^{(m)} - mu_n||^2 = ||eps_m||^2 / (n - m)^2
and the conservation identity (18):
    n mu_n = sum_{remaining} Phi + sum_{discarded} Phi.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def g_remaining_mean(phi, r, mu=None):
    """Mean embedding of X without x_r: g(x_r) = (n mu_n - Phi(x_r)) / (n-1).

    Equations (12) and the display above it.
    """
    n = phi.shape[0]
    if mu is None:
        mu = jnp.mean(phi, axis=0)
    return (n * mu - phi[r]) / (n - 1)


def j_removal(phi, r, mu=None):
    """J(x_r) = ||mu_n - g(x_r)||^2 = 1/(n-1)^2 ||Phi(x_r) - mu_n||^2.

    Equation (12). Returns (J, scaled_sq), which must match.
    """
    n = phi.shape[0]
    if mu is None:
        mu = jnp.mean(phi, axis=0)
    g = g_remaining_mean(phi, r, mu=mu)
    diff_mu_g = mu - g
    j_val = jnp.sum(diff_mu_g * diff_mu_g)
    diff_phi = phi[r] - mu
    scaled = jnp.sum(diff_phi * diff_phi) / ((n - 1) ** 2)
    return j_val, scaled


def backward_kernel_herding(phi, target_size: int):
    """Algorithm 6: Backward Kernel Herding via mean embedding.

    Args:
        phi: (n, q) explicit feature matrix, rows Phi(x_i).
        target_size: T, number of points to keep (T < n).

    Returns:
        remaining: (T,) indices of X^{(n-T)}.
        removed: (n-T,) indices in removal order (R_{n-T}).
        eps: residual eps_{n-T} in R^q.
        mu: empirical mean embedding mu_n.
    """
    phi = jnp.asarray(phi)
    n = phi.shape[0]
    n_remove = n - int(target_size)
    if n_remove < 0:
        raise ValueError("target_size cannot exceed n")
    mu = jnp.mean(phi, axis=0)

    def body(state, _):
        active, eps = state
        # Alg 6 line 5: argmin ||Phi(x_r) - mu_n - eps||^2 over remaining r.
        diff = phi - mu[None, :] - eps[None, :]
        scores = jnp.sum(diff * diff, axis=1)
        scores = jnp.where(active, scores, jnp.inf)
        r = jnp.argmin(scores)
        # Alg 6 line 6 / eq (15): eps <- Phi(x_rm) - mu_n + eps.
        eps_new = phi[r] - mu + eps
        active_new = active.at[r].set(False)
        return (active_new, eps_new), r

    active0 = jnp.ones((n,), dtype=bool)
    eps0 = jnp.zeros((phi.shape[1],), dtype=phi.dtype)
    (active, eps), removed = jax.lax.scan(body, (active0, eps0), xs=None, length=n_remove)
    remaining = jnp.nonzero(active, size=int(target_size), fill_value=-1)[0]
    return remaining, removed, eps, mu


def kernelized_backward_herding(k, target_size: int):
    """Algorithm 7: Kernelized Backward Herding via mean embedding.

    Precomputes K_mu(x) = (1/n) sum_i k(x, x_i) and I(x) <- K_mu(x).
    Each step removes argmin (k(x, x) - 2 I(x)) and updates
        I(x) <- I(x) + k(x, x_rm) - K_mu(x)
    for remaining x.

    Args:
        k: (n, n) kernel matrix.
        target_size: T, number of points to keep.

    Returns:
        remaining: (T,) indices.
        removed: (n-T,) indices in removal order.
    """
    k = jnp.asarray(k)
    n = k.shape[0]
    n_remove = n - int(target_size)
    if n_remove < 0:
        raise ValueError("target_size cannot exceed n")
    k_mu = jnp.mean(k, axis=1)
    diag = jnp.diag(k)

    def body(state, _):
        active, i_vec = state
        scores = diag - 2.0 * i_vec
        scores = jnp.where(active, scores, jnp.inf)
        r = jnp.argmin(scores)
        i_new = i_vec + k[:, r] - k_mu
        active_new = active.at[r].set(False)
        return (active_new, i_new), r

    active0 = jnp.ones((n,), dtype=bool)
    (active, _), removed = jax.lax.scan(body, (active0, k_mu), xs=None, length=n_remove)
    remaining = jnp.nonzero(active, size=int(target_size), fill_value=-1)[0]
    return remaining, removed
