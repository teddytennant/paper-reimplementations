"""Poisson control variates, TGI-RWM summands, SNIS, and pruning."""

from __future__ import annotations

from tgi_cv._host import ensure_nix_ld

ensure_nix_ld()

import jax
import jax.numpy as jnp

from tgi_cv.hermite import evaluate_modes
from tgi_cv.kernel import beta_from_gamma, mode_eigenvalues, tgi_rwm_step

# Estimator coefficients. Not (-1, -1).
THETA = (1.0, -1.0)


def poisson_eta(c, lam):
    """eta_j = c_j / (1 - lambda_j)."""
    return jnp.asarray(c) / (1.0 - jnp.asarray(lam))


def g_value(z, modes, eta):
    """G = sum_j eta_j H_j."""
    z = jnp.asarray(z)
    modes = list(modes)
    if not modes:
        return jnp.zeros(z.shape[:-1], dtype=z.dtype)
    return evaluate_modes(modes, z) @ jnp.asarray(eta, dtype=z.dtype)


def qg_value(z, modes, eta, lam):
    """QG(z) = sum_j lambda_j eta_j phi_j(z)."""
    z = jnp.asarray(z)
    modes = list(modes)
    if not modes:
        return jnp.zeros(z.shape[:-1], dtype=z.dtype)
    coeff = jnp.asarray(lam, dtype=z.dtype) * jnp.asarray(eta, dtype=z.dtype)
    return evaluate_modes(modes, z) @ coeff


def control_terms(z, z_prop, modes, eta, lam, alpha):
    """H1 = alpha*(G(z')-G(z)), H2 = G(z') - sum lambda_j eta_j phi_j(z)."""
    g = g_value(z, modes, eta)
    g_prop = g_value(z_prop, modes, eta)
    qg = qg_value(z, modes, eta, lam)
    h1 = alpha * (g_prop - g)
    h2 = g_prop - qg
    return h1, h2, g, qg


def summand(f_tz, z, z_prop, modes, eta, lam, alpha, theta=THETA):
    """S = F(T(z)) + theta1*H1 + theta2*H2."""
    h1, h2, _, _ = control_terms(z, z_prop, modes, eta, lam, alpha)
    theta1, theta2 = theta
    return f_tz + theta1 * h1 + theta2 * h2


def degree1_modes(d):
    return [tuple(1 if i == j else 0 for i in range(d)) for j in range(d)]


def affine_linear_coefficients(L, weight):
    """Hermite coefficients of z |-> weight · L z. Mean weight · b is separate."""
    weight = jnp.asarray(weight)
    return jnp.asarray(L).T @ weight


def snis(w, f_vals, phi, a, b):
    """Self-normalized weights with dictionary controls.

    Numerator is sum_i w_i F(T(zeta_i)) - sum_i sum_j a_j phi_j(zeta_i).
    Denominator is sum_i w_i - sum_i sum_j b_j phi_j(zeta_i).
    a_j is the coefficient of w*(F o T) on phi_j, b_j of w.
    """
    w = jnp.asarray(w)
    f_vals = jnp.asarray(f_vals)
    phi = jnp.asarray(phi)
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    numer = jnp.sum(w * f_vals - phi @ a)
    denom = jnp.sum(w - phi @ b)
    return numer / denom


def hermite_coefficients(values, phi):
    """Monte Carlo coefficients and sample standard deviations of the terms."""
    terms = jnp.asarray(values)[..., None] * jnp.asarray(phi)
    c_hat = jnp.mean(terms, axis=0)
    s_hat = jnp.std(terms, axis=0, ddof=1)
    return c_hat, s_hat


def prune_mask(c_hat, s_hat, tau, n0):
    """Keep j if c_hat_j**2 > tau * s_hat_j**2 / N0."""
    c_hat = jnp.asarray(c_hat)
    s_hat = jnp.asarray(s_hat)
    return jnp.square(c_hat) > (tau * jnp.square(s_hat) / n0)


def run_tgi_rwm(key, z0, log_w_fn, f_fn, gamma, n_steps, modes, eta, theta=THETA):
    """Run TGI-RWM and return summands, plain values, accept bits, and alpha."""
    z0 = jnp.asarray(z0)
    eta = jnp.asarray(eta, dtype=z0.dtype)
    modes = [tuple(int(v) for v in k) for k in modes]
    beta = beta_from_gamma(gamma)
    lam = mode_eigenvalues(modes, jnp.asarray(beta, dtype=z0.dtype))

    def body(carry, step_key):
        z, log_w = carry
        z_next, log_w_next, z_prop, alpha, accept, _ = tgi_rwm_step(
            step_key, z, log_w, log_w_fn, gamma
        )
        f_val = jnp.asarray(f_fn(z), dtype=z.dtype)
        s = summand(f_val, z, z_prop, modes, eta, lam, alpha, theta)
        return (z_next, log_w_next), (s, f_val, accept, alpha)

    log_w0 = jnp.asarray(log_w_fn(z0), dtype=z0.dtype)
    keys = jax.random.split(key, n_steps)
    _, out = jax.lax.scan(body, (z0, log_w0), keys)
    return out
