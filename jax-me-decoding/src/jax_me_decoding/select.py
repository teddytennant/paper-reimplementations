"""Greedy ME-Decoding: Mahalanobis energy, rank-1 Cholesky, truncation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from jax_me_decoding.kernel import adaptive_bandwidth, rbf_kernel

Array = jax.Array

DEFAULT_LAMBDA = 0.9
DEFAULT_TOP_N = 512
SCHUR_EPS = 1e-8


def mahalanobis_energy(p_s: Array, k_ss: Array) -> Array:
    """F(S) = p_S^T K_{S,S}^{-1} p_S, computed from scratch."""
    p_s = jnp.atleast_1d(jnp.asarray(p_s))
    k_ss = jnp.atleast_2d(jnp.asarray(k_ss))
    x = jnp.linalg.solve(k_ss, p_s)
    return jnp.dot(p_s, x)


def _as_1d(logits: Array) -> Array:
    logits = jnp.asarray(logits)
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D (V,), got shape {logits.shape}")
    return logits


def _as_embeddings(embeddings: Array, vocab: int) -> Array:
    embeddings = jnp.asarray(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings[:, None]
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (V, d), got shape {embeddings.shape}")
    if int(embeddings.shape[0]) != vocab:
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != logits length ({vocab})"
        )
    return embeddings


def _energy_of_subset(p: Array, k: Array, selected: list[int]) -> Array:
    idx = jnp.asarray(selected, dtype=jnp.int32)
    p_s = p[idx]
    k_ss = k[idx[:, None], idx]
    return mahalanobis_energy(p_s, k_ss)


def _greedy_reference(
    p: Array, k: Array, lambda_: float
) -> tuple[list[int], list[Array]]:
    n = int(p.shape[0])
    diag = jnp.diag(k)
    j0 = int(jnp.argmax((p * p) / diag))
    selected = [j0]
    f = _energy_of_subset(p, k, selected)
    f_path = [f]
    p_c_sq = jnp.sum(p * p)
    in_s = [False] * n
    in_s[j0] = True

    while len(selected) < n:
        best_j = -1
        best_f = None
        for j in range(n):
            if in_s[j]:
                continue
            f_j = _energy_of_subset(p, k, selected + [j])
            if best_j < 0 or f_j > best_f:
                best_f = f_j
                best_j = j
        a_s = jnp.sum(p[jnp.asarray(selected)] ** 2) / p_c_sq
        if (f > 0) and ((best_f - f) / f < lambda_ * (1.0 - a_s)):
            break
        selected.append(best_j)
        in_s[best_j] = True
        f = best_f
        f_path.append(f)
    return selected, f_path


def _greedy_cholesky(
    p: Array, k: Array, lambda_: float
) -> tuple[list[int], list[Array]]:
    n = int(p.shape[0])
    diag = jnp.diag(k)
    j0 = int(jnp.argmax((p * p) / diag))
    selected = [j0]
    remaining = jnp.ones((n,), dtype=bool).at[j0].set(False)

    l00 = jnp.sqrt(diag[j0])
    L = l00.reshape(1, 1)
    w = (p[j0] / l00).reshape(1)
    f = w[0] * w[0]
    f_path = [f]
    p_c_sq = jnp.sum(p * p)

    while len(selected) < n:
        s_idx = jnp.asarray(selected, dtype=jnp.int32)
        k_s_all = k[s_idx]  # (t, n)
        v = jsp_linalg.solve_triangular(L, k_s_all, lower=True)
        schur = diag - jnp.sum(v * v, axis=0)
        residual = p - w @ v
        gain = jnp.where(
            schur > SCHUR_EPS,
            (residual * residual) / jnp.maximum(schur, SCHUR_EPS),
            jnp.zeros_like(schur),
        )
        gain = jnp.where(remaining, gain, jnp.full_like(gain, -jnp.inf))
        j_star = int(jnp.argmax(gain))
        f_new = f + jnp.where(jnp.isfinite(gain[j_star]), gain[j_star], 0.0)
        a_s = jnp.sum(p[s_idx] ** 2) / p_c_sq
        if (f > 0) and ((f_new - f) / f < lambda_ * (1.0 - a_s)):
            break
        if schur[j_star] <= SCHUR_EPS:
            break
        v_j = v[:, j_star]
        ell = jnp.sqrt(schur[j_star])
        t = L.shape[0]
        L_new = jnp.zeros((t + 1, t + 1), dtype=L.dtype)
        L_new = L_new.at[:t, :t].set(L)
        L_new = L_new.at[t, :t].set(v_j)
        L_new = L_new.at[t, t].set(ell)
        L = L_new
        w = jnp.concatenate([w, (residual[j_star] / ell).reshape(1)])
        f = f_new
        selected.append(j_star)
        remaining = remaining.at[j_star].set(False)
        f_path.append(f)
    return selected, f_path


def greedy_select(
    p: Array,
    k: Array,
    lambda_: float = DEFAULT_LAMBDA,
    method: str = "cholesky",
) -> tuple[Array, Array, dict[str, Any]]:
    """Greedy maximisation of F(S) = p_S^T K_{S,S}^{-1} p_S on a pool.

    Parameters
    ----------
    p :
        Candidate-pool probabilities, shape (n,).
    k :
        Kernel on the same pool, shape (n, n).
    lambda_ :
        Early-stop parameter λ. Stop when
        (F(S∪{j*}) - F(S)) / F(S) < λ (1 - A_S),
        A_S = ||p_S||_2^2 / ||p||_2^2.
    method :
        ``"cholesky"`` uses the rank-1 Cholesky update; ``"reference"``
        inverts K_{S,S} from scratch at every candidate (tiny n only).

    Returns
    -------
    selected :
        Indices into the pool, in greedy order.
    f_path :
        F(S) after each accepted token.
    info :
        Diagnostics (method, first token, A_S of the final S).
    """
    p = jnp.atleast_1d(jnp.asarray(p))
    k = jnp.atleast_2d(jnp.asarray(k))
    if method == "reference":
        selected, f_path = _greedy_reference(p, k, float(lambda_))
    elif method == "cholesky":
        selected, f_path = _greedy_cholesky(p, k, float(lambda_))
    else:
        raise ValueError(f"unknown method {method!r}")
    sel = jnp.asarray(selected, dtype=jnp.int32)
    path = jnp.stack(f_path) if f_path else jnp.zeros((0,), dtype=p.dtype)
    p_c_sq = jnp.sum(p * p)
    a_s = jnp.sum(p[sel] ** 2) / p_c_sq
    info = {
        "method": method,
        "first_token": int(sel[0]) if sel.size else -1,
        "A_S": a_s,
        "pool_size": int(p.shape[0]),
    }
    return sel, path, info


def me_select(
    logits: Array,
    embeddings: Array,
    lambda_: float = DEFAULT_LAMBDA,
    temperature: float = 1.0,
    top_n: int = DEFAULT_TOP_N,
    method: str = "cholesky",
) -> tuple[Array, Array, dict[str, Any]]:
    """Select a token support from logits and embeddings.

    Returns ``(selected_indices into V, F_path, info)``.
    """
    logits = _as_1d(logits)
    embeddings = _as_embeddings(embeddings, int(logits.shape[0]))
    p_full = jax.nn.softmax(logits / temperature)
    vocab = int(p_full.shape[0])
    n_pool = min(int(top_n), vocab)
    candidates = jnp.argsort(p_full)[::-1][:n_pool]
    p = p_full[candidates]
    e_c = embeddings[candidates]
    eps = adaptive_bandwidth(e_c)
    k = rbf_kernel(e_c, eps)
    selected_in_pool, f_path, ginfo = greedy_select(p, k, lambda_=lambda_, method=method)
    selected = candidates[selected_in_pool]
    info = {
        **ginfo,
        "candidates": candidates,
        "selected_in_pool": selected_in_pool,
        "epsilon": eps,
        "lambda": float(lambda_),
        "temperature": float(temperature),
        "top_n": n_pool,
    }
    return selected, f_path, info


def me_truncate_logits(
    logits: Array,
    embeddings: Array,
    lambda_: float = DEFAULT_LAMBDA,
    temperature: float = 1.0,
    top_n: int = DEFAULT_TOP_N,
    method: str = "cholesky",
) -> Array:
    """Keep original logits on the selected support; set the rest to -inf."""
    logits = _as_1d(logits)
    selected, _, _ = me_select(
        logits,
        embeddings,
        lambda_=lambda_,
        temperature=temperature,
        top_n=top_n,
        method=method,
    )
    out = jnp.full_like(logits, -jnp.inf)
    return out.at[selected].set(logits[selected])


def me_sample(
    key: Array,
    logits: Array,
    embeddings: Array,
    lambda_: float = DEFAULT_LAMBDA,
    temperature: float = 1.0,
    top_n: int = DEFAULT_TOP_N,
    method: str = "cholesky",
) -> Array:
    """Sample one token from the truncated (renormalised on S) distribution."""
    trunc = me_truncate_logits(
        logits,
        embeddings,
        lambda_=lambda_,
        temperature=temperature,
        top_n=top_n,
        method=method,
    )
    return jax.random.categorical(key, trunc)
