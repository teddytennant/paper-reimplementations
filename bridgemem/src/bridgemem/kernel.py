"""Displayed BridgeMem equations (arXiv:2609.29268 eqs. 1–5)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from bridgemem.constants import EPS, HIDDEN, K, LN_EPS, N_BINS, TAU


def recency_bin(delta_t):
    """Map lag Δt to a bin index using edges {0, 1, 4, 16, 64}."""
    delta_t = jnp.asarray(delta_t)
    # searchsorted on (1, 4, 16, 64) with side='left':
    # Δt <= 1 -> 0, <= 4 -> 1, <= 16 -> 2, <= 64 -> 3, else 4.
    cuts = jnp.array([1, 4, 16, 64], dtype=delta_t.dtype)
    return jnp.clip(jnp.searchsorted(cuts, delta_t, side="left"), 0, N_BINS - 1)


def init_params(key, n_rel, dim=64, hidden=HIDDEN, n_bins=N_BINS):
    """Event embeddings, attention vector w, LN, and 128-unit transition head."""
    n_dir = 2 * n_rel
    keys = jax.random.split(key, 6)
    return {
        "e_r": jax.random.normal(keys[0], (n_rel, dim)) * 0.02,
        "e_d": jax.random.normal(keys[1], (2, dim)) * 0.02,
        "e_b": jax.random.normal(keys[2], (n_bins, dim)) * 0.02,
        "w": jax.random.normal(keys[3], (dim,)) * 0.02,
        "ln_scale": jnp.ones((dim,)),
        "ln_bias": jnp.zeros((dim,)),
        "W1": jax.random.normal(keys[4], (dim, hidden)) * jnp.sqrt(2.0 / dim),
        "b1": jnp.zeros((hidden,)),
        "W2": jax.random.normal(keys[5], (hidden, n_dir)) * jnp.sqrt(2.0 / hidden),
        "b2": jnp.zeros((n_dir,)),
    }


def event_embedding(params, relation, direction, bins):
    """eq. 1: z_j = e_{r_j} + e_{d_j} + e_{b(Δt_j)}."""
    return (
        params["e_r"][relation]
        + params["e_d"][direction]
        + params["e_b"][bins]
    )


def attention_pool(params, z, mask):
    """eq. 1: α_j = softmax_j(w^T tanh z_j), m = Σ_j α_j z_j.

    `mask` is True for events in B_{a,c,t}. Padded slots get score -1e9.
    """
    scores = jnp.einsum("...kd,d->...k", jnp.tanh(z), params["w"])
    scores = jnp.where(mask, scores, jnp.array(-1.0e9, dtype=scores.dtype))
    alpha = jax.nn.softmax(scores, axis=-1)
    m = jnp.einsum("...k,...kd->...d", alpha, z)
    return alpha, m


def layer_norm(x, scale, bias, eps=LN_EPS):
    """LN(m) in eq. 2, last axis."""
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return scale * (x - mu) * jax.lax.rsqrt(var + eps) + bias


def mlp_head(params, m):
    """128-unit transition head: LN(m) then ReLU hidden then logits."""
    h = layer_norm(m, params["ln_scale"], params["ln_bias"])
    h = jax.nn.relu(h @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def neural_log_probs(params, m, log_pi):
    """eq. 2: p_θ(r|B) = softmax(MLP(LN(m)) + log π)_r as log-probabilities."""
    logits = mlp_head(params, m) + log_pi
    return jax.nn.log_softmax(logits, axis=-1)


def A_nn(log_p, log_pi, target):
    """eq. 2: A_nn = log p_θ(r|B) - log π_r."""
    return log_p[..., target] - log_pi[..., target]


def A_ct(p_ctx, log_pi_r, mask):
    """eq. 3: A_ct = log Σ_{j ∈ B} p_ct(r | d_j, r_j, b_j) - log π_r.

    `p_ctx` is p_ct(r | token_j) for each retrieved event, same shape as `mask`.
    """
    p_sum = jnp.sum(jnp.where(mask, p_ctx, 0.0), axis=-1)
    return jnp.log(p_sum) - log_pi_r


def rho_B(n_B, tau=TAU):
    """eq. 4: ρ_B = τ / (n_B + τ), τ=100."""
    n_B = jnp.asarray(n_B)
    return tau / (n_B + tau)


def mix_weight_count(n_B, tau=TAU):
    """Weight (1-ρ_B) on the count estimator."""
    return 1.0 - rho_B(n_B, tau)


def mixed_residual(A_count, A_neural, n_B, tau=TAU):
    """eq. 4: A(q,c) = log[(1-ρ_B) e^{A_ct} + ρ_B e^{A_nn}]."""
    rho = rho_B(n_B, tau)
    return jnp.log((1.0 - rho) * jnp.exp(A_count) + rho * jnp.exp(A_neural))


def g_gate(u_q):
    """eq. 5: g(q) = 4 u_q (1 - u_q)."""
    return 4.0 * u_q * (1.0 - u_q)


def final_log_score(p0, A, u_q, lam=1.0, eps=EPS):
    """eq. 5: S(q,c) = log max{p_0(c|q), ε} + λ g(q) A(q,c), ε=10^{-12}."""
    return jnp.log(jnp.maximum(p0, eps)) + lam * g_gate(u_q) * A


def n_B_from_tokens(token_n, direction, relation, bins, mask):
    """n_B: total training count supporting the retrieved transition tokens."""
    n_j = token_n[direction, relation, bins]
    return jnp.sum(jnp.where(mask, n_j, 0.0), axis=-1)


def p_ct_from_tokens(p_ct_table, direction, relation, bins, target, mask):
    """Look up p_ct(r | d_j, r_j, b_j) for each event j."""
    p_j = p_ct_table[direction, relation, bins, target]
    return jnp.where(mask, p_j, 0.0)


def residual_from_tokens(
    params,
    relation,
    direction,
    bins,
    mask,
    p_ct_table,
    token_n,
    log_pi,
    target,
    tau=TAU,
):
    """A(q,c) for one or a batch of candidate token lists. Empty B => 0."""
    has_hist = jnp.any(mask, axis=-1)
    z = event_embedding(params, relation, direction, bins)
    _, m = attention_pool(params, z, mask)
    log_p = neural_log_probs(params, m, log_pi)
    a_nn = A_nn(log_p, log_pi, target)
    p_ctx = p_ct_from_tokens(p_ct_table, direction, relation, bins, target, mask)
    a_ct = A_ct(p_ctx, log_pi[target], mask)
    n_b = n_B_from_tokens(token_n, direction, relation, bins, mask)
    a = mixed_residual(a_ct, a_nn, n_b, tau=tau)
    return jnp.where(has_hist, a, jnp.zeros_like(a))


def log_scores(
    params,
    relation,
    direction,
    bins,
    mask,
    p_ct_table,
    token_n,
    log_pi,
    target,
    p0,
    u_q,
    lam=1.0,
    tau=TAU,
    eps=EPS,
):
    """eq. 5 log scores: backbone log p0 plus gated residual."""
    A = residual_from_tokens(
        params,
        relation,
        direction,
        bins,
        mask,
        p_ct_table,
        token_n,
        log_pi,
        target,
        tau=tau,
    )
    return final_log_score(p0, A, u_q, lam=lam, eps=eps), A


def pad_K(relation, direction, bins, mask=None, k=K):
    """Pad a variable-length token list to K slots."""
    n = int(relation.shape[0])
    if mask is None:
        mask = jnp.ones((n,), dtype=bool)
    if n >= k:
        return relation[:k], direction[:k], bins[:k], mask[:k]
    pad = k - n

    def _pad(x, fill):
        return jnp.concatenate([x, jnp.full((pad,), fill, dtype=x.dtype)])

    return (
        _pad(relation, 0),
        _pad(direction, 0),
        _pad(bins, 0),
        jnp.concatenate([mask, jnp.zeros((pad,), dtype=bool)]),
    )
