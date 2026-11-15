"""Q-DEQ inner loop and Anderson mixing baseline (eq 2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from q_deq.deq import head, shared_layer
from q_deq.implicit import make_deq_rewire
from q_deq.quantize import quantized_shared_layer
from q_deq.qubo import (
    candidate_directions,
    decode_coefficients,
    directional_p,
    encode_qubo,
    gram_h,
    merge_qubo_diagonal,
)
from q_deq.sa import sa_solve


def anderson_mix(z_hist, f_hist, beta=0.8, ridge=1e-6):
    """Eq 2: z' = (1-β) z^{(k)} + β sum_j α_j F(z^{(k-m+j)}, x).

    α solves min ||R α||^2 s.t. 1^T α = 1, R rows = residuals f - z.
    """
    r_hist = f_hist - z_hist
    G = r_hist @ r_hist.T
    m1 = G.shape[0]
    G = G + ridge * jnp.eye(m1, dtype=G.dtype)
    A = jnp.zeros((m1 + 1, m1 + 1), dtype=G.dtype)
    A = A.at[:m1, :m1].set(G)
    A = A.at[:m1, m1].set(1.0)
    A = A.at[m1, :m1].set(1.0)
    rhs = jnp.zeros((m1 + 1,), dtype=G.dtype).at[m1].set(1.0)
    sol = jnp.linalg.solve(A, rhs)
    alpha = sol[:m1]
    f_mix = alpha @ f_hist
    z_new = (1.0 - beta) * z_hist[-1] + beta * f_mix
    return z_new, alpha


def anderson_solve(F, z0, x, n_iter=12, m=3, beta=0.8):
    """Anderson accelerated fixed-point iteration (comparison baseline)."""
    z = z0
    zs = [z]
    fs = [F(z, x)]
    for _ in range(n_iter):
        take = min(m + 1, len(zs))
        z_hist = jnp.stack(zs[-take:])
        f_hist = jnp.stack(fs[-take:])
        z, _ = anderson_mix(z_hist, f_hist, beta=beta)
        zs.append(z)
        fs.append(F(z, x))
    return z


def qdeq_step(
    F,
    z,
    x,
    hist,
    key,
    n_dirs=4,
    eta=1e-3,
    n_sweeps=80,
    n_restarts=2,
    t0=1.0,
    t_min=1e-3,
):
    """One local QUBO step: directions -> eq 3–6 -> SA -> z + V a."""
    r = F(z, x) - z
    key, k_dir, k_sa = jax.random.split(key, 3)
    V = candidate_directions(z, r, hist, k_dir, n_dirs)
    P = directional_p(F, z, x, V, eta=eta)
    G, h = gram_h(P, r)
    Q, q, _const, _B, _offset = encode_qubo(G, h, r)
    Q, q = merge_qubo_diagonal(Q, q)
    x_bits, _e = sa_solve(
        Q, q, k_sa, n_sweeps=n_sweeps, t0=t0, t_min=t_min, n_restarts=n_restarts
    )
    a = decode_coefficients(x_bits, n_dirs, dtype=z.dtype)
    z_new = z + V @ a
    return z_new


def qdeq_solve(
    F,
    z0,
    x,
    key,
    k_inner=8,
    n_dirs=4,
    history_len=3,
    eta=1e-3,
    n_sweeps=80,
    n_restarts=2,
):
    """K_inner local discrete QUBO updates. No gradient through SA."""
    z = z0
    hist = jnp.broadcast_to(z0[None, :], (history_len, z0.shape[-1]))

    def body(_i, carry):
        z, hist, key = carry
        key, k1 = jax.random.split(key)
        z_new = qdeq_step(
            F,
            z,
            x,
            hist,
            k1,
            n_dirs=n_dirs,
            eta=eta,
            n_sweeps=n_sweeps,
            n_restarts=n_restarts,
        )
        hist = jnp.concatenate([hist[1:], z[None, :]], axis=0)
        return z_new, hist, key

    z, hist, key = jax.lax.fori_loop(0, k_inner, body, (z, hist, key))
    return z


def _quant_F(params, z, x):
    return quantized_shared_layer(params, z, x)


deq_rewire_w8a8 = make_deq_rewire(_quant_F)


def training_forward(
    layer_params,
    head_params,
    x,
    key,
    z0=None,
    k_inner=6,
    n_dirs=3,
    n_sweeps=40,
    n_restarts=1,
    use_w8a8=True,
):
    """Algorithm 1: stop-grad Q-DEQ solve, W8A8 re-forward, implicit VJP.

    The value sent to the head is F_W8A8(z*, x). Backward uses a dense
    solve of (I - ∂F/∂z)^T, not SA unrolling.
    """
    if z0 is None:
        z0 = jnp.zeros((layer_params["Wz"].shape[0],), dtype=x.dtype)

    def F_fp(z, xx):
        return shared_layer(layer_params, z, xx)

    z_star = qdeq_solve(
        F_fp,
        z0,
        x,
        key,
        k_inner=k_inner,
        n_dirs=n_dirs,
        n_sweeps=n_sweeps,
        n_restarts=n_restarts,
    )
    z_star = jax.lax.stop_gradient(z_star)
    if use_w8a8:
        z_out = deq_rewire_w8a8(layer_params, z_star, x)
    else:
        z_out = shared_layer(layer_params, z_star, x)
    return head(head_params, z_out)
