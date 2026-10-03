"""Schur Replay scale selection (section 4.1 and Appendix Algorithm 1).

Ambiguities resolved from the appendix pseudocode and section 4.1.3:
- A nonpositive fixed-code update is clamped to SCALE_FLOOR (1e-30).
- If Q^T S_G Q <= 1e-12, the current iterate s^{(t)} is kept (Algorithm 1).
- Loss ties within TIE_TOL of the minimum prefer the projected old scale,
  then the smaller scale.
- Pi_S midpoint ties take the smaller scale.
- The window center is numpy searchsorted(S, s*, side='left'), clipped to the grid.
"""

from functools import partial

import jax
import jax.numpy as jnp

from schur_replay.nvfp4 import rho

GROUP_SIZE = 16
DEFAULT_HALF_WIDTH = 4
DEFAULT_ITERATIONS = 3
DENOM_TOL = 1e-12
DIAG_EPS = 1e-12
DEFAULT_TIE_TOL = 1e-8
SCALE_FLOOR = 1e-30


def inverse_upper_factor(H):
    """Upper-triangular U with H^{-1} = U.T @ U.

    This is the Cholesky factor of the inverse used in equation 3.
    """
    H = 0.5 * (H + H.T)
    gram_inv = jnp.linalg.inv(H)
    gram_inv = 0.5 * (gram_inv + gram_inv.T)
    return jnp.linalg.cholesky(gram_inv, upper=True)


def schur_from_factor(U_GG):
    """S_G = (U_GG.T @ U_GG)^{-1} (equation 8)."""
    gram = U_GG.T @ U_GG
    metric = jnp.linalg.inv(gram)
    return 0.5 * (metric + metric.T)


def schur_complement(H, g_start, g_end):
    """S_G = H_GG - H_GF inv(H_FF) H_FG (equation 7).

    Group columns are [g_start, g_end) and the future is [g_end, n).
    This form has no committed prefix. After a prefix is fixed, pass the
    trailing conditional Hessian, or use schur_from_factor on U_GG.
    """
    H = 0.5 * (H + H.T)
    block = H[g_start:g_end, g_start:g_end]
    n = H.shape[0]
    if g_end >= n:
        return block
    cross = H[g_start:g_end, g_end:]
    future = H[g_end:, g_end:]
    metric = block - cross @ jnp.linalg.inv(future) @ cross.T
    return 0.5 * (metric + metric.T)


def propose_scales(w0, metric, s0, iterations=DEFAULT_ITERATIONS, denom_tol=DENOM_TOL):
    """J fixed-code iterations from the incoming scale (equations 10-11).

    w0 is (M, B), metric is (B, B), s0 is (M,). Returns the continuous
    proposal, shape (M,). A row whose denominator is at most denom_tol keeps
    its current iterate. A valid nonpositive update is clamped to SCALE_FLOOR.
    """
    w0 = jnp.asarray(w0)
    metric = jnp.asarray(metric, dtype=w0.dtype)
    s = jnp.asarray(s0, dtype=w0.dtype)
    if s.ndim != 1 or s.shape[0] != w0.shape[0]:
        raise ValueError("s0 must have shape (M,)")
    floor = jnp.asarray(SCALE_FLOOR, dtype=w0.dtype)
    tol = jnp.asarray(denom_tol, dtype=w0.dtype)
    for _ in range(iterations):
        divisor = jnp.where(s > 0, s, floor)
        codes = rho(w0 / divisor[:, None])
        weighted = codes @ metric
        denom = jnp.sum(weighted * codes, axis=-1)
        numer = jnp.sum(weighted * w0, axis=-1)
        valid = denom > tol
        updated = numer / jnp.maximum(denom, tol)
        updated = jnp.where(updated > 0, updated, floor)
        s = jnp.where(valid, updated, s)
    return s


def replay_group(w0, U_GG, scales, diag_eps=DIAG_EPS):
    """Recurrence-exact replay (equations 12-13).

    Every candidate starts from the same w0. State is private per candidate.
    w0 is (M, B). scales is (M, K) or (M,). Returns q of shape (M, K, B).
    The diagonal guard max(u_jj, diag_eps) matches Algorithm 2 and is a no-op
    for ordinary Cholesky diagonals.
    """
    w0 = jnp.asarray(w0)
    U_GG = jnp.asarray(U_GG, dtype=w0.dtype)
    scales = jnp.asarray(scales, dtype=w0.dtype)
    if scales.ndim == 1:
        scales = scales[:, None]
    if scales.shape[0] != w0.shape[0]:
        raise ValueError("scales must have one row per output row")
    if U_GG.shape != (w0.shape[1], w0.shape[1]):
        raise ValueError("U_GG must be (B, B)")
    return _replay_compiled(w0, U_GG, scales, jnp.asarray(diag_eps, dtype=w0.dtype))


@jax.jit
def _replay_compiled(w0, U_GG, scales, diag_eps):
    rows, width = w0.shape
    n_cand = scales.shape[1]
    state = jnp.broadcast_to(w0[:, None, :], (rows, n_cand, width))

    def body(state, j):
        current = state[:, :, j]
        qj = scales * rho(current / scales)
        diagonal = jnp.maximum(U_GG[j, j], diag_eps)
        error = (current - qj) / diagonal
        # Columns k <= j stay put. Later columns take the GPTQ step.
        mask = (jnp.arange(width) > j).astype(state.dtype)
        state = state - error[:, :, None] * U_GG[j] * mask
        return state, qj

    _state, codes = jax.lax.scan(body, state, jnp.arange(width))
    return jnp.moveaxis(codes, 0, -1)


def conditional_score(delta, metric):
    """L = delta^T S_G delta. delta is (M, K, B) or (M, B); returns (M, K) or (M,)."""
    delta = jnp.asarray(delta)
    metric = jnp.asarray(metric, dtype=delta.dtype)
    squeezed = delta.ndim == 2
    if squeezed:
        delta = delta[:, None, :]
    transformed = jnp.matmul(delta, metric)
    loss = jnp.sum(transformed * delta, axis=-1)
    if squeezed:
        return loss[:, 0]
    return loss


def project_to_grid(grid, values):
    """Nearest exportable scale. Midpoint ties take the smaller scale.

    grid is sorted ascending. values broadcasts over the last axis lookup.
    Returns (projected values, indices).
    """
    grid = jnp.asarray(grid)
    values = jnp.asarray(values, dtype=grid.dtype)
    right = jnp.searchsorted(grid, values, side="left")
    right = jnp.clip(right, 0, grid.shape[0] - 1)
    left = jnp.clip(right - 1, 0, grid.shape[0] - 1)
    left_val = grid[left]
    right_val = grid[right]
    take_left = jnp.abs(values - left_val) <= jnp.abs(values - right_val)
    index = jnp.where(take_left, left, right)
    return grid[index], index


def candidate_window(grid, proposal, s_old, half_width=DEFAULT_HALF_WIDTH):
    """Local exportable window around the proposal, plus Pi_S(s_old) (equation 14).

    Returns candidates (M, 2h+2) and the projected old scale (M,).
    The last column is always Pi_S(s_old). Boundary clipping may repeat scales.
    """
    grid = jnp.asarray(grid)
    proposal = jnp.asarray(proposal, dtype=grid.dtype)
    s_old = jnp.asarray(s_old, dtype=grid.dtype)
    count = grid.shape[0]
    projected, _ = project_to_grid(grid, s_old)
    center = jnp.searchsorted(grid, proposal, side="left")
    center = jnp.clip(center, 0, count - 1)
    offsets = jnp.arange(-half_width, half_width + 1)
    index = jnp.clip(center[:, None] + offsets[None, :], 0, count - 1)
    window = grid[index]
    candidates = jnp.concatenate([window, projected[:, None]], axis=1)
    return candidates, projected


def _pick_winner(losses, candidates, tie_tol):
    """Prefer the old scale when it is within tie_tol of the minimum, else the smaller scale.

    The old scale occupies the last candidate column.
    """
    minimum = jnp.min(losses, axis=1)
    near = losses <= (minimum[:, None] + tie_tol)
    old_near = near[:, -1]
    masked = jnp.where(near, candidates, jnp.inf)
    smallest = jnp.argmin(masked, axis=1)
    old_index = candidates.shape[1] - 1
    winner_index = jnp.where(old_near, old_index, smallest)
    winner = jnp.take_along_axis(candidates, winner_index[:, None], axis=1)[:, 0]
    return winner, winner_index


def select_scales(
    w0,
    U_GG,
    metric,
    s_old,
    grid,
    half_width=DEFAULT_HALF_WIDTH,
    iterations=DEFAULT_ITERATIONS,
    tie_tol=DEFAULT_TIE_TOL,
    denom_tol=DENOM_TOL,
):
    """Propose, replay, and pick one exportable scale per row (Algorithm 1, before commit).

    Returns the winning scales (M,) and a dict of proposal, projected old scale,
    candidates, losses, and replayed codes.
    """
    w0 = jnp.asarray(w0)
    U_GG = jnp.asarray(U_GG, dtype=w0.dtype)
    metric = jnp.asarray(metric, dtype=w0.dtype)
    s_old = jnp.asarray(s_old, dtype=w0.dtype)
    grid = jnp.asarray(grid, dtype=w0.dtype)
    rows, width = w0.shape
    if U_GG.shape != (width, width) or metric.shape != (width, width):
        raise ValueError("U_GG and metric must be (B, B)")
    if s_old.shape != (rows,):
        raise ValueError("s_old must have shape (M,)")
    if grid.ndim != 1:
        raise ValueError("grid must be 1-D and sorted ascending")
    proposal = propose_scales(w0, metric, s_old, iterations=iterations, denom_tol=denom_tol)
    candidates, projected = candidate_window(grid, proposal, s_old, half_width=half_width)
    codes = replay_group(w0, U_GG, candidates)
    delta = codes - w0[:, None, :]
    losses = conditional_score(delta, metric)
    winner, winner_index = _pick_winner(losses, candidates, tie_tol)
    return winner, {
        "proposal": proposal,
        "projected_old": projected,
        "candidates": candidates,
        "losses": losses,
        "codes": codes,
        "winner_index": winner_index,
    }


def commit_group(W, U, g_start, scales, group_size):
    """Commit the winning scale with the GPTQ column update (Algorithm 1, last loop).

    Quantized columns are written back. The error is compensated into later columns.
    """
    W = jnp.asarray(W)
    U = jnp.asarray(U, dtype=W.dtype)
    scales = jnp.asarray(scales, dtype=W.dtype)
    if scales.shape != (W.shape[0],):
        raise ValueError("scales must have shape (M,)")
    return _commit_compiled(W, U, int(g_start), scales, int(group_size))


@partial(jax.jit, static_argnums=(4,))
def _commit_compiled(W, U, g_start, scales, group_size):
    cols = W.shape[1]
    diag_eps = jnp.asarray(DIAG_EPS, dtype=W.dtype)
    start = jnp.asarray(g_start, dtype=jnp.int32)

    def body(W, j):
        col = start + j
        current = W[:, col]
        quantized = scales * rho(current / scales)
        diagonal = jnp.maximum(U[col, col], diag_eps)
        error = (current - quantized) / diagonal
        W = W.at[:, col].set(quantized)
        mask = (jnp.arange(cols) > col).astype(W.dtype)
        W = W - error[:, None] * U[col] * mask
        return W, None

    updated, _unused = jax.lax.scan(body, W, jnp.arange(group_size))
    return updated
