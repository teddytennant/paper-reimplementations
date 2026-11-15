"""Refinement variable xi, LSTM unroll, and first-order meta-training."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from psagml.lstm import LSTMParams, init_lstm, init_lstm_state, lstm_step, preprocess_grad
from psagml.pso import random_phase_zf_wsr, run_pso
from psagml.star_ris import (
    composite_channel,
    phi_to_tilde,
    star_coefficients,
    unconstrained_phi,
)
from psagml.wsr import weighted_sum_rate
from psagml.zf import collapsed_precoder, power_normalize, zf_collapsed_X


def pack_xi(phi_tilde, theta_t, theta_r, X):
    """xi = [phi_tilde; theta_t; theta_r; vec_real_imag(X)], D = 3N + 2 K^2."""
    x_ri = jnp.concatenate([jnp.real(X).reshape(-1), jnp.imag(X).reshape(-1)])
    return jnp.concatenate([phi_tilde, theta_t, theta_r, x_ri])


def unpack_xi(xi, N, K):
    phi_tilde = xi[:N]
    theta_t = xi[N : 2 * N]
    theta_r = xi[2 * N : 3 * N]
    rest = xi[3 * N :]
    kk = K * K
    X = rest[:kk].reshape(K, K) + 1j * rest[kk:].reshape(K, K)
    return phi_tilde, theta_t, theta_r, X


def reconstruct_A_W(xi, G, Ht, Hr, Pmax):
    """Rebuild A and power-normalized W = A^H X from xi."""
    N = G.shape[0]
    K = Ht.shape[0] + Hr.shape[0]
    phi_tilde, theta_t, theta_r, X = unpack_xi(xi, N, K)
    phi = unconstrained_phi(phi_tilde)
    vt, vr = star_coefficients(phi, theta_t, theta_r)
    A = composite_channel(Ht, Hr, G, vt, vr)
    W = power_normalize(collapsed_precoder(A, X), Pmax)
    return A, W


def wsr_from_xi(xi, G, Ht, Hr, Pmax, sigma2, omega):
    A, W = reconstruct_A_W(xi, G, Ht, Hr, Pmax)
    return weighted_sum_rate(A, W, omega, sigma2)


def pso_to_xi(x_pso, G, Ht, Hr):
    """Warm-start xi from PSO gbest: unconstrained phi, ZF-collapsed X."""
    N = G.shape[0]
    phi = x_pso[:N]
    theta_t = x_pso[N : 2 * N]
    theta_r = x_pso[2 * N : 3 * N]
    vt, vr = star_coefficients(phi, theta_t, theta_r)
    A = composite_channel(Ht, Hr, G, vt, vr)
    X = zf_collapsed_X(A)
    return pack_xi(phi_to_tilde(phi), theta_t, theta_r, X)


def _two_lstm_step(Mc, Mx, g, states, N, nu):
    n_c = 3 * N
    g_c, g_x = g[:n_c], g[n_c:]
    hc, cc, hx, cx = states
    hc, cc, dc = lstm_step(Mc, g_c, hc, cc, nu)
    hx, cx, dx = lstm_step(Mx, g_x, hx, cx, nu)
    delta = jnp.concatenate([dc, dx])
    return (hc, cc, hx, cx), delta


def refine(Mc, Mx, xi0, G, Ht, Hr, Pmax, sigma2, omega, S, nu, hidden):
    """S LSTM steps. Retain the best iterate (Alg. 1 / Prop 3).

    Gradients fed to the LSTM are stop-gradient (first-order L2O).
    Returns (xi_best, R_best, trajectory of WSR after each step).
    """
    N = G.shape[0]
    K = Ht.shape[0] + Hr.shape[0]
    n_c = 3 * N
    n_x = 2 * K * K
    hc, cc = init_lstm_state(n_c, hidden, dtype=xi0.dtype)
    hx, cx = init_lstm_state(n_x, hidden, dtype=xi0.dtype)
    states = (hc, cc, hx, cx)

    R0 = wsr_from_xi(xi0, G, Ht, Hr, Pmax, sigma2, omega)
    xi = xi0
    xi_best = xi0
    R_best = R0
    traj = []

    def value_and_grad_R(z):
        return jax.value_and_grad(wsr_from_xi)(z, G, Ht, Hr, Pmax, sigma2, omega)

    for _ in range(S):
        R, grad = value_and_grad_R(xi)
        better = R > R_best
        R_best = jnp.where(better, R, R_best)
        xi_best = jnp.where(better, xi, xi_best)
        g = preprocess_grad(jax.lax.stop_gradient(grad))
        states, delta = _two_lstm_step(Mc, Mx, g, states, N, nu)
        xi = xi + delta
        R_after = wsr_from_xi(xi, G, Ht, Hr, Pmax, sigma2, omega)
        traj.append(R_after)

    R_final = wsr_from_xi(xi, G, Ht, Hr, Pmax, sigma2, omega)
    better = R_final > R_best
    R_best = jnp.where(better, R_final, R_best)
    xi_best = jnp.where(better, xi, xi_best)
    traj_arr = jnp.stack(traj) if traj else jnp.array([R0])
    return xi_best, R_best, traj_arr


def unroll_wsrs(Mc, Mx, xi0, G, Ht, Hr, Pmax, sigma2, omega, T, nu, hidden):
    """WSR after each of T LSTM updates (for loss 31)."""
    N = G.shape[0]
    K = Ht.shape[0] + Hr.shape[0]
    hc, cc = init_lstm_state(3 * N, hidden, dtype=xi0.dtype)
    hx, cx = init_lstm_state(2 * K * K, hidden, dtype=xi0.dtype)
    states = (hc, cc, hx, cx)
    xi = xi0
    wsrs = []

    def grad_R(z):
        return jax.grad(wsr_from_xi)(z, G, Ht, Hr, Pmax, sigma2, omega)

    for _ in range(T):
        grad = grad_R(xi)
        g = preprocess_grad(jax.lax.stop_gradient(grad))
        states, delta = _two_lstm_step(Mc, Mx, g, states, N, nu)
        xi = xi + delta
        wsrs.append(wsr_from_xi(xi, G, Ht, Hr, Pmax, sigma2, omega))
    return jnp.stack(wsrs)


def meta_loss(psi, batch, T, nu, hidden, Pmax, sigma2):
    """Eq (31): negative mean WSR over unrolled steps and channel draws.

    psi = (Mc, Mx). batch is a list of (G, Ht, Hr, xi0, omega).
    """
    Mc, Mx = psi
    total = 0.0
    n = 0
    for G, Ht, Hr, xi0, omega in batch:
        wsrs = unroll_wsrs(Mc, Mx, xi0, G, Ht, Hr, Pmax, sigma2, omega, T, nu, hidden)
        total = total + (-jnp.mean(wsrs))
        n += 1
    return total / jnp.maximum(n, 1)


def _adam_init(params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v


def _adam_update(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * (g * g), v, grads)
    mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, m, v


def meta_train(
    key,
    channels,
    T=3,
    n_steps=4,
    hidden=8,
    nu=0.05,
    lr=1e-3,
    Pmax=10.0,
    sigma2=1.0,
    n_particles=4,
    n_pso_iters=5,
    warm_start="pso",
):
    """A few Adam steps on psi. channels: list of (G, Ht, Hr).

    Returns (psi, losses) with psi = (Mc, Mx).
    """
    k_c, k_x, key = jax.random.split(key, 3)
    Mc = init_lstm(k_c, hidden)
    Mx = init_lstm(k_x, hidden)
    psi = (Mc, Mx)

    batch = []
    for G, Ht, Hr in channels:
        K = Ht.shape[0] + Hr.shape[0]
        omega = jnp.ones((K,), dtype=jnp.float32)
        key, k_i = jax.random.split(key)
        if warm_start == "pso":
            x_best, _, _ = run_pso(
                k_i, G, Ht, Hr, Pmax, sigma2, omega, n_particles, n_pso_iters
            )
            xi0 = pso_to_xi(x_best, G, Ht, Hr)
        else:
            N = G.shape[0]
            xi0 = jax.random.normal(k_i, (3 * N + 2 * K * K,)) * 0.1
        batch.append((G, Ht, Hr, xi0, omega))

    def loss_fn(p):
        return meta_loss(p, batch, T, nu, hidden, Pmax, sigma2)

    m, v = _adam_init(psi)
    losses = []
    for t in range(1, n_steps + 1):
        loss, grads = jax.value_and_grad(loss_fn)(psi)
        losses.append(loss)
        psi, m, v = _adam_update(psi, grads, m, v, t, lr=lr)
    loss_final = loss_fn(psi)
    losses.append(loss_final)
    return psi, jnp.stack(losses)


def run_psagml(
    key,
    G,
    Ht,
    Hr,
    Pmax=10.0,
    sigma2=1.0,
    omega=None,
    n_particles=4,
    n_pso_iters=5,
    hidden=8,
    nu=0.05,
    S=2,
    Mc: LSTMParams | None = None,
    Mx: LSTMParams | None = None,
):
    """PSO warm start then S LSTM refinement steps on one channel."""
    K = Ht.shape[0] + Hr.shape[0]
    if omega is None:
        omega = jnp.ones((K,), dtype=jnp.float32)
    k_pso, k_lstm = jax.random.split(key)
    x_best, wsr_pso, hist = run_pso(
        k_pso, G, Ht, Hr, Pmax, sigma2, omega, n_particles, n_pso_iters
    )
    if Mc is None or Mx is None:
        k1, k2 = jax.random.split(k_lstm)
        Mc = init_lstm(k1, hidden)
        Mx = init_lstm(k2, hidden)
    xi0 = pso_to_xi(x_best, G, Ht, Hr)
    xi_best, wsr_best, traj = refine(
        Mc, Mx, xi0, G, Ht, Hr, Pmax, sigma2, omega, S, nu, hidden
    )
    return {
        "wsr_pso": wsr_pso,
        "wsr_refined": wsr_best,
        "xi": xi_best,
        "pso_hist": hist,
        "traj": traj,
        "x_pso": x_best,
        "Mc": Mc,
        "Mx": Mx,
    }


def random_baseline(key, G, Ht, Hr, Pmax, sigma2, omega):
    return random_phase_zf_wsr(key, G, Ht, Hr, Pmax, sigma2, omega)
