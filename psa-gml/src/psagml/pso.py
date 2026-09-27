"""PSO warm start over STAR-RIS coefficients (eqs 18-23)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from psagml.star_ris import composite_channel, star_coefficients
from psagml.wsr import weighted_sum_rate
from psagml.zf import power_normalize, zf_precoder


def project_search_space(x, N):
    """Pi_S: clip phi to [0, pi/2], wrap phases to [0, 2pi)."""
    phi = jnp.clip(x[:N], 0.0, jnp.pi / 2.0)
    theta_t = jnp.mod(x[N : 2 * N], 2.0 * jnp.pi)
    theta_r = jnp.mod(x[2 * N : 3 * N], 2.0 * jnp.pi)
    return jnp.concatenate([phi, theta_t, theta_r])


def zf_wsr_from_x(x, G, Ht, Hr, Pmax, sigma2, omega):
    """Fitness (eq 19): WSR of power-normalized ZF at A(x)."""
    N = G.shape[0]
    phi = x[:N]
    theta_t = x[N : 2 * N]
    theta_r = x[2 * N : 3 * N]
    vt, vr = star_coefficients(phi, theta_t, theta_r)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W = power_normalize(zf_precoder(A), Pmax)
    return weighted_sum_rate(A, W, omega, sigma2)


def random_phase_zf_wsr(key, G, Ht, Hr, Pmax, sigma2, omega):
    """Single random point in S, then normalized ZF WSR."""
    N = G.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    phi = jax.random.uniform(k1, (N,), minval=0.0, maxval=jnp.pi / 2.0)
    theta_t = jax.random.uniform(k2, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    theta_r = jax.random.uniform(k3, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    x = jnp.concatenate([phi, theta_t, theta_r])
    return zf_wsr_from_x(x, G, Ht, Hr, Pmax, sigma2, omega)


def run_pso(
    key,
    G,
    Ht,
    Hr,
    Pmax,
    sigma2,
    omega,
    n_particles=4,
    n_iters=5,
    inertia=0.7,
    c1=1.5,
    c2=1.5,
):
    """PSO on x = [phi; theta_t; theta_r] with ZF fitness.

    Returns (gbest_x, gbest_fitness, gbest_history) where history is
    the global-best fitness after initialization and after each iteration
    (nondecreasing by eqs 22-23).
    """
    N = G.shape[0]
    dim = 3 * N
    k_x, k_v, key = jax.random.split(key, 3)
    phi = jax.random.uniform(k_x, (n_particles, N), minval=0.0, maxval=jnp.pi / 2.0)
    # reuse split for phases
    k_t, k_r = jax.random.split(k_v)
    theta_t = jax.random.uniform(k_t, (n_particles, N), minval=0.0, maxval=2.0 * jnp.pi)
    theta_r = jax.random.uniform(k_r, (n_particles, N), minval=0.0, maxval=2.0 * jnp.pi)
    x = jnp.concatenate([phi, theta_t, theta_r], axis=1)
    v = jnp.zeros((n_particles, dim), dtype=x.dtype)

    def fit_one(xi):
        return zf_wsr_from_x(xi, G, Ht, Hr, Pmax, sigma2, omega)

    fit_all = jax.vmap(fit_one)
    pfit = fit_all(x)
    p = x
    g_idx = jnp.argmax(pfit)
    g = p[g_idx]
    gfit = pfit[g_idx]
    hist = [gfit]

    def project_all(xs):
        return jax.vmap(lambda xi: project_search_space(xi, N))(xs)

    for _ in range(n_iters):
        key, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.uniform(k1, x.shape)
        r2 = jax.random.uniform(k2, x.shape)
        v = inertia * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
        x = project_all(x + v)
        f = fit_all(x)
        improved = f > pfit
        p = jnp.where(improved[:, None], x, p)
        pfit = jnp.where(improved, f, pfit)
        g_idx = jnp.argmax(pfit)
        g = p[g_idx]
        gfit = pfit[g_idx]
        hist.append(gfit)

    return g, gfit, jnp.stack(hist)
