"""Dense and tiled FlashBoB (Algorithm 2 of arXiv 2609.24089).

Single-head attention with tau = 1/sqrt(d):

    S = tau Q K^T + Mmask,  P = softmax_row(S),  O = P V
    L_i = logsumexp(S_i) over valid columns
    dP = dO V^T,  D_i = dO_i · O_i,  dS = P * (dP - D[:, None])
    dQ = tau dS K,  dK = tau dS^T Q,  dV = P^T dO

Phi = <dQ, UQ> + <dK, UK> + <dV, UV>, with the Frobenius inner product.
The outputs are the gradient of Phi with respect to Q, K, V, and dO.

A boolean mask uses True for a valid (kept) score. Invalid scores are -inf
and drop out of the softmax and of every row sum. causal_mask(n) keeps
column j for row i when j <= i. Block sizes Br and Bc are the pass-1 row
block and the pass-2 column block. The same sizes tile the streamed axis.
N need not be divisible by either size. The tiled schedule stores no N x N
tensor: every interaction matrix exists only as a Br x Bc tile.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class FlashBoBResult(NamedTuple):
    """BoB outputs plus the pass-1 accumulators used by the affine fusion.

    The first four fields are dQ, dK, dV, ddO, so result[:4] is the gradient.
    traffic is the Appendix H element count for the tiled schedule, or None
    for the dense reference.
    """

    dQ: jnp.ndarray
    dK: jnp.ndarray
    dV: jnp.ndarray
    ddO: jnp.ndarray
    alpha: jnp.ndarray
    E: jnp.ndarray
    E_base: jnp.ndarray
    Omega: jnp.ndarray
    B: jnp.ndarray
    R: jnp.ndarray
    ddO_base: jnp.ndarray
    O: jnp.ndarray
    L: jnp.ndarray
    D: jnp.ndarray
    traffic: Optional[int]


def causal_mask(n: int) -> jnp.ndarray:
    """Boolean mask of shape (n, n). True where column j is visible to row i (j <= i)."""
    if n < 1:
        raise ValueError("n must be positive")
    idx = jnp.arange(n)
    return idx[None, :] <= idx[:, None]


def _tau(d: int) -> float:
    return 1.0 / math.sqrt(d)


def _prepare_arrays(Q, K, V, dO, UQ, UK, UV):
    dtype = jnp.float64
    arrays = [jnp.asarray(a, dtype=dtype) for a in (Q, K, V, dO, UQ, UK, UV)]
    n, d = arrays[0].shape
    if arrays[0].ndim != 2 or n < 1 or d < 1:
        raise ValueError("Q must have shape (N, d) with N >= 1 and d >= 1")
    for a in arrays:
        if a.shape != (n, d):
            raise ValueError(f"expected every matrix to have shape {(n, d)}, got {a.shape}")
    return (*arrays, n, d)


def _prepare_mask(mask, n: int, causal: bool):
    if mask is None and not causal:
        return None
    if causal:
        base = causal_mask(n)
    else:
        base = None
    if mask is None:
        return base
    mask = jnp.asarray(mask, dtype=bool)
    if mask.shape != (n, n):
        raise ValueError(f"mask must have shape {(n, n)}, got {mask.shape}")
    if base is None:
        return mask
    return base & mask


def _blocks(n: int, block: int):
    if block < 1:
        raise ValueError("block size must be positive")
    start = 0
    while start < n:
        end = min(start + block, n)
        yield start, end
        start = end


def _as_mask(mask, n: int):
    """Validate a caller-supplied mask. Does not build a causal matrix."""
    if mask is None:
        return None
    mask = jnp.asarray(mask, dtype=bool)
    if mask.shape != (n, n):
        raise ValueError(f"mask must have shape {(n, n)}, got {mask.shape}")
    return mask


def _valid_tile(mask, causal: bool, i0, i1, j0, j1):
    """Boolean tile. Causal visibility is j <= i, computed from indices, not a stored N x N mask."""
    br = i1 - i0
    bc = j1 - j0
    if mask is None and not causal:
        return jnp.ones((br, bc), dtype=bool)
    valid = jnp.ones((br, bc), dtype=bool)
    if causal:
        rows = jnp.arange(i0, i1)[:, None]
        cols = jnp.arange(j0, j1)[None, :]
        valid = cols <= rows
    if mask is not None:
        valid = valid & mask[i0:i1, j0:j1]
    return valid


def forward_attention(Q, K, V, mask=None, causal: bool = False):
    """Softmax attention. Returns O, P, L.

    L_i is logsumexp over valid columns. Invalid mask entries are -inf and
    have probability 0. Each row must contain at least one valid column.
    """
    Q = jnp.asarray(Q, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    V = jnp.asarray(V, dtype=jnp.float64)
    n, d = Q.shape
    if K.shape != (n, d) or V.shape != (n, d):
        raise ValueError("Q, K, V must share shape (N, d)")
    mask = _prepare_mask(mask, n, causal)
    tau = _tau(d)
    S = tau * (Q @ K.T)
    if mask is not None:
        S = jnp.where(mask, S, -jnp.inf)
    L = jax.nn.logsumexp(S, axis=-1)
    P = jax.nn.softmax(S, axis=-1)
    O = P @ V
    return O, P, L


def attention(Q, K, V, mask=None, causal: bool = False):
    """Forward softmax attention, returning only O. This is the map differentiated by flashbob_autodiff."""
    O, _, _ = forward_attention(Q, K, V, mask=mask, causal=causal)
    return O


def _row_state(Q, K, V, dO, mask):
    """Saved forward and first-backward row state: O, L, D. D_i = dO_i · O_i."""
    n, d = Q.shape
    tau = _tau(d)
    S = tau * (Q @ K.T)
    if mask is not None:
        S = jnp.where(mask, S, -jnp.inf)
    L = jax.nn.logsumexp(S, axis=-1)
    P = jax.nn.softmax(S, axis=-1)
    O = P @ V
    D = jnp.sum(dO * O, axis=-1)
    return O, L, D


def _probs(Qi, Kj, Li, tau, valid):
    """Recompute the softmax tile from the saved logsumexp, as in Algorithm 2."""
    S = tau * (Qi @ Kj.T)
    S = jnp.where(valid, S, -jnp.inf)
    P = jnp.exp(S - Li[:, None])
    return jnp.where(jnp.isfinite(P), P, jnp.zeros_like(P))


@jax.jit
def _pass1_tile(Qi, dOi, UQi, Li, Di, Kj, Vj, UKj, UVj, tau, valid):
    """Alpha-free pass-1 contributions of one tile. Eq. (14) and eq. (32)."""
    P = _probs(Qi, Kj, Li, tau, valid)
    dP = dOi @ Vj.T
    F = tau * (UQi @ Kj.T + Qi @ UKj.T)
    C = dOi @ UVj.T
    delta = dP - Di[:, None]
    dP_base = C + F * delta
    alpha = jnp.sum(P * F, axis=-1)
    E_base = jnp.sum(P * dP_base, axis=-1)
    ddO_base = P @ UVj + (P * F) @ Vj
    Omega = (P * dP_base) @ Kj + (P * delta) @ UKj
    B = P @ Kj
    R = (P * dP) @ Kj
    return alpha, E_base, ddO_base, Omega, B, R


@jax.jit
def _pass2_tile(Qi, dOi, UQi, Li, Di, alpha_i, E_i, Kj, Vj, UKj, UVj, tau, valid):
    """Column-owned increments of dK and dV for one tile."""
    P = _probs(Qi, Kj, Li, tau, valid)
    dP = dOi @ Vj.T
    F = tau * (UQi @ Kj.T + Qi @ UKj.T)
    C = dOi @ UVj.T
    delta = dP - Di[:, None]
    h = F - alpha_i[:, None]
    dS = P * delta
    dP_bob = C + F * delta - alpha_i[:, None] * dP
    dS_bob = P * (dP_bob - E_i[:, None])
    dV = (P * h).T @ dOi
    dK = tau * (dS_bob.T @ Qi + dS.T @ UQi)
    return dK, dV


def _finalize_rows(alpha, E_base, ddO_base, Omega, B, R, D, O, tau):
    """Affine fusion, eq. (8): fold alpha into E, ddO, and dQ."""
    E = E_base - alpha * D
    ddO = ddO_base - alpha[:, None] * O
    dQ = tau * (Omega - alpha[:, None] * R - E[:, None] * B)
    return E, ddO, dQ


def _result(dQ, dK, dV, ddO, alpha, E, E_base, Omega, B, R, ddO_base, O, L, D, traffic):
    return FlashBoBResult(
        dQ, dK, dV, ddO, alpha, E, E_base, Omega, B, R, ddO_base, O, L, D, traffic
    )


def flashbob_dense(Q, K, V, dO, UQ, UK, UV, mask=None, causal: bool = False) -> FlashBoBResult:
    """Dense reference. Materializing P is fine here; the tiled path never does."""
    Q, K, V, dO, UQ, UK, UV, n, d = _prepare_arrays(Q, K, V, dO, UQ, UK, UV)
    mask = _prepare_mask(mask, n, causal)
    tau = _tau(d)
    O, L, D = _row_state(Q, K, V, dO, mask)
    valid = jnp.ones((n, n), dtype=bool) if mask is None else mask
    alpha, E_base, ddO_base, Omega, B, R = _pass1_tile(
        Q, dO, UQ, L, D, K, V, UK, UV, tau, valid
    )
    E, ddO, dQ = _finalize_rows(alpha, E_base, ddO_base, Omega, B, R, D, O, tau)
    dK, dV = _pass2_tile(Q, dO, UQ, L, D, alpha, E, K, V, UK, UV, tau, valid)
    return _result(
        dQ, dK, dV, ddO, alpha, E, E_base, Omega, B, R, ddO_base, O, L, D, None
    )


class _Ledger:
    """Appendix H element traffic. Counts loads and stores, not JAX internals.

    Pass 1, once per row block: load Qi, dOi, UQi, Oi, Li, Di.
    Pass 1, once per tile: load Kj, Vj, UKj, UVj.
    Pass 1, once per row block: store dQ_i, ddO_i, alpha_i, E_i.
    Pass 2, once per column block: load Kj, Vj, UKj, UVj.
    Pass 2, once per tile: load Qi, dOi, UQi, Li, Di, alpha_i, E_i.
    Pass 2, once per column block: store dK_j, dV_j.
    O is not loaded in pass 2. The mask is not in the ledger.
    """

    def __init__(self) -> None:
        self.elements = 0

    def add(self, n: int) -> None:
        self.elements += int(n)


def _row_state_tiled(Q, K, V, dO, mask, causal: bool, Br: int, Bc: int):
    """O, L, D without storing an N x N score matrix.

    Online softmax over Br x Bc tiles. L and O match the dense row state up
    to float64 summation order. D_i = dO_i · O_i, the statistic Algorithm 2 loads.
    """
    n, d = Q.shape
    tau = _tau(d)
    O = jnp.zeros((n, d), dtype=Q.dtype)
    L = jnp.zeros((n,), dtype=Q.dtype)
    for i0, i1 in _blocks(n, Br):
        Qi = Q[i0:i1]
        br = i1 - i0
        m_i = jnp.full((br,), -jnp.inf, dtype=Q.dtype)
        ell_i = jnp.zeros((br,), dtype=Q.dtype)
        acc_i = jnp.zeros((br, d), dtype=Q.dtype)
        for j0, j1 in _blocks(n, Bc):
            valid = _valid_tile(mask, causal, i0, i1, j0, j1)
            S = tau * (Qi @ K[j0:j1].T)
            S = jnp.where(valid, S, -jnp.inf)
            block_max = jnp.max(S, axis=-1)
            m_new = jnp.maximum(m_i, block_max)
            scale = jnp.exp(m_i - m_new)
            scale = jnp.where(jnp.isfinite(scale), scale, jnp.zeros_like(scale))
            p = jnp.exp(S - m_new[:, None])
            p = jnp.where(jnp.isfinite(p), p, jnp.zeros_like(p))
            ell_i = scale * ell_i + jnp.sum(p, axis=-1)
            acc_i = scale[:, None] * acc_i + p @ V[j0:j1]
            m_i = m_new
        safe = ell_i > 0
        Li = jnp.where(safe, m_i + jnp.log(ell_i), -jnp.inf)
        Oi = jnp.where(safe[:, None], acc_i / jnp.where(safe, ell_i, 1.0)[:, None], 0.0)
        L = L.at[i0:i1].set(Li)
        O = O.at[i0:i1].set(Oi)
    D = jnp.sum(dO * O, axis=-1)
    return O, L, D


def flashbob_tiled(
    Q,
    K,
    V,
    dO,
    UQ,
    UK,
    UV,
    Br: int,
    Bc: int,
    mask=None,
    causal: bool = False,
) -> FlashBoBResult:
    """Algorithm 2. Br is the pass-1 row block, Bc the pass-2 column block.

    Both passes stream the other axis at the same block size. Partial tail
    blocks are kept when N is not divisible by Br or Bc. No N x N tensor is
    stored: scores, probabilities, F, C, and dS exist only as tiles.
    """
    Q, K, V, dO, UQ, UK, UV, n, d = _prepare_arrays(Q, K, V, dO, UQ, UK, UV)
    Br = int(Br)
    Bc = int(Bc)
    if Br < 1 or Bc < 1:
        raise ValueError("Br and Bc must be positive")
    mask = _as_mask(mask, n)
    tau = _tau(d)
    # Row state is an input of Algorithm 2. Compute it here so the schedule
    # can be called with the seven matrices alone. This is not BoB traffic.
    # Causal entries are filled per tile, so this function stores no N x N tensor.
    O, L, D = _row_state_tiled(Q, K, V, dO, mask, causal, Br, Bc)

    dQ = jnp.zeros((n, d), dtype=Q.dtype)
    ddO = jnp.zeros((n, d), dtype=Q.dtype)
    alpha = jnp.zeros((n,), dtype=Q.dtype)
    E = jnp.zeros((n,), dtype=Q.dtype)
    E_base = jnp.zeros((n,), dtype=Q.dtype)
    Omega = jnp.zeros((n, d), dtype=Q.dtype)
    B = jnp.zeros((n, d), dtype=Q.dtype)
    R = jnp.zeros((n, d), dtype=Q.dtype)
    ddO_base = jnp.zeros((n, d), dtype=Q.dtype)
    ledger = _Ledger()

    # Pass 1, row-major. Finalize alpha, E, dQ, ddO.
    for i0, i1 in _blocks(n, Br):
        br = i1 - i0
        Qi = Q[i0:i1]
        dOi = dO[i0:i1]
        UQi = UQ[i0:i1]
        Oi = O[i0:i1]
        Li = L[i0:i1]
        Di = D[i0:i1]
        # Stationary loads: Qi, dOi, UQi, Oi, Li, Di.
        ledger.add(4 * br * d + 2 * br)

        alpha_i = jnp.zeros((br,), dtype=Q.dtype)
        E_base_i = jnp.zeros((br,), dtype=Q.dtype)
        ddO_base_i = jnp.zeros((br, d), dtype=Q.dtype)
        Omega_i = jnp.zeros((br, d), dtype=Q.dtype)
        B_i = jnp.zeros((br, d), dtype=Q.dtype)
        R_i = jnp.zeros((br, d), dtype=Q.dtype)

        for j0, j1 in _blocks(n, Bc):
            bc = j1 - j0
            Kj = K[j0:j1]
            Vj = V[j0:j1]
            UKj = UK[j0:j1]
            UVj = UV[j0:j1]
            # Streamed loads: Kj, Vj, UKj, UVj. Reloaded for every row block.
            ledger.add(4 * bc * d)
            valid = _valid_tile(mask, causal, i0, i1, j0, j1)
            a, e0, do0, om, b, r = _pass1_tile(
                Qi, dOi, UQi, Li, Di, Kj, Vj, UKj, UVj, tau, valid
            )
            alpha_i = alpha_i + a
            E_base_i = E_base_i + e0
            ddO_base_i = ddO_base_i + do0
            Omega_i = Omega_i + om
            B_i = B_i + b
            R_i = R_i + r

        E_i, ddO_i, dQ_i = _finalize_rows(
            alpha_i, E_base_i, ddO_base_i, Omega_i, B_i, R_i, Di, Oi, tau
        )
        # Stores: dQ_i, ddO_i, alpha_i, E_i. On-chip accumulators are not written.
        ledger.add(2 * br * d + 2 * br)
        alpha = alpha.at[i0:i1].set(alpha_i)
        E = E.at[i0:i1].set(E_i)
        E_base = E_base.at[i0:i1].set(E_base_i)
        dQ = dQ.at[i0:i1].set(dQ_i)
        ddO = ddO.at[i0:i1].set(ddO_i)
        ddO_base = ddO_base.at[i0:i1].set(ddO_base_i)
        Omega = Omega.at[i0:i1].set(Omega_i)
        B = B.at[i0:i1].set(B_i)
        R = R.at[i0:i1].set(R_i)

    # Pass 2, column-major. Recompute tiles from L and the saved (alpha, E).
    dK = jnp.zeros((n, d), dtype=Q.dtype)
    dV = jnp.zeros((n, d), dtype=Q.dtype)
    for j0, j1 in _blocks(n, Bc):
        bc = j1 - j0
        Kj = K[j0:j1]
        Vj = V[j0:j1]
        UKj = UK[j0:j1]
        UVj = UV[j0:j1]
        # Stationary loads: Kj, Vj, UKj, UVj.
        ledger.add(4 * bc * d)
        dK_j = jnp.zeros((bc, d), dtype=Q.dtype)
        dV_j = jnp.zeros((bc, d), dtype=Q.dtype)

        for i0, i1 in _blocks(n, Br):
            br = i1 - i0
            Qi = Q[i0:i1]
            dOi = dO[i0:i1]
            UQi = UQ[i0:i1]
            Li = L[i0:i1]
            Di = D[i0:i1]
            alpha_i = alpha[i0:i1]
            E_i = E[i0:i1]
            # Streamed loads: Qi, dOi, UQi, Li, Di, alpha_i, E_i. O is not loaded.
            ledger.add(3 * br * d + 4 * br)
            valid = _valid_tile(mask, causal, i0, i1, j0, j1)
            dk, dv = _pass2_tile(
                Qi, dOi, UQi, Li, Di, alpha_i, E_i, Kj, Vj, UKj, UVj, tau, valid
            )
            dK_j = dK_j + dk
            dV_j = dV_j + dv

        # Stores: dK_j, dV_j.
        ledger.add(2 * bc * d)
        dK = dK.at[j0:j1].set(dK_j)
        dV = dV.at[j0:j1].set(dV_j)

    return _result(
        dQ, dK, dV, ddO, alpha, E, E_base, Omega, B, R, ddO_base, O, L, D, ledger.elements
    )


def flashbob_autodiff(Q, K, V, dO, UQ, UK, UV, mask=None, causal: bool = False):
    """jax.grad of Phi through a VJP of forward attention.

    The inner VJP maps dO to (dQ, dK, dV). Phi is the Frobenius pairing of
    those cotangents with (UQ, UK, UV). The returned tuple is the gradient
    of Phi with respect to (Q, K, V, dO).
    """
    Q, K, V, dO, UQ, UK, UV, n, _ = _prepare_arrays(Q, K, V, dO, UQ, UK, UV)
    mask = _prepare_mask(mask, n, causal)

    def phi(Q, K, V, dO):
        def fwd(Q, K, V):
            return attention(Q, K, V, mask=mask)

        _, vjp = jax.vjp(fwd, Q, K, V)
        dq, dk, dv = vjp(dO)
        return jnp.sum(dq * UQ) + jnp.sum(dk * UK) + jnp.sum(dv * UV)

    return jax.grad(phi, argnums=(0, 1, 2, 3))(Q, K, V, dO)
