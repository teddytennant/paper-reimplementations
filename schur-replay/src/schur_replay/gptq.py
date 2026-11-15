"""Row-wise GPTQ driver. Each group picks a scale with Schur Replay, then compensates forward.

Algorithms 3-5 (memory tiering, checkpoints, tensor-parallel row splits) are not implemented.
Column order is the order of W and H. No activation reordering is applied.
"""

import jax.numpy as jnp

from schur_replay.nvfp4 import E2M1_MAX, exportable_scales
from schur_replay.schur import (
    GROUP_SIZE,
    DEFAULT_HALF_WIDTH,
    DEFAULT_ITERATIONS,
    DEFAULT_TIE_TOL,
    commit_group,
    inverse_upper_factor,
    project_to_grid,
    schur_from_factor,
    select_scales,
)

DEFAULT_DAMPING = 0.01


def initial_group_scales(w0, grid):
    """Per-row RTN scale of the group at entry: project(amax / 6) onto the grid."""
    row_amax = jnp.max(jnp.abs(w0), axis=1)
    raw = row_amax / jnp.asarray(E2M1_MAX, dtype=w0.dtype)
    projected, _ = project_to_grid(grid, raw)
    return projected


def damped_inverse_factor(H, damping=DEFAULT_DAMPING, max_tries=8):
    """Symmetrize H, damp by `damping` times the mean diagonal, then factor.

    Failed Cholesky attempts increase the damping. An unresolved failure raises
    rather than substituting an identity Hessian.
    """
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    eye = jnp.eye(n, dtype=H.dtype)
    mean_diag = jnp.mean(jnp.diag(H))
    base = jnp.where(mean_diag > 0, mean_diag, jnp.asarray(1.0, dtype=H.dtype))
    damp = float(damping)
    for _ in range(max_tries):
        damped = H + (damp * base) * eye
        factor = inverse_upper_factor(damped)
        if bool(jnp.all(jnp.isfinite(factor))):
            return factor
        damp *= 10.0
    raise RuntimeError("Cholesky of the inverse Hessian failed after increasing damping")


def gptq_quantize(
    W,
    H=None,
    X=None,
    group_size=GROUP_SIZE,
    half_width=DEFAULT_HALF_WIDTH,
    iterations=DEFAULT_ITERATIONS,
    damping=DEFAULT_DAMPING,
    tie_tol=DEFAULT_TIE_TOL,
    s0=None,
):
    """Quantize W left to right, one NVFP4 group at a time.

    W is (M, N). Pass H = X X^T of shape (N, N), or X of shape (N, T).
    N must be divisible by group_size. The exportable grid is frozen from the
    original tensor amax (equation 27). If s0 is omitted, each group's incoming
    scale is the entry-time per-row amax/6 projected onto that grid.

    Returns (quantized weight, scales) with scales shaped (M, N / group_size).
    Scales are effective multipliers s, not raw E4M3 codes.
    """
    W = jnp.asarray(W)
    if H is None:
        if X is None:
            raise ValueError("pass H or X")
        X = jnp.asarray(X, dtype=W.dtype)
        H = X @ X.T
    else:
        H = jnp.asarray(H, dtype=W.dtype)
    if W.ndim != 2:
        raise ValueError("W must have shape (M, N)")
    rows, cols = W.shape
    if group_size <= 0 or cols % group_size != 0:
        raise ValueError("N must be divisible by group_size")
    if H.shape != (cols, cols):
        raise ValueError("H must have shape (N, N)")
    amax = jnp.max(jnp.abs(W))
    if not bool(amax > 0):
        raise ValueError("weight tensor amax must be positive")
    factor = damped_inverse_factor(H, damping=damping)
    grid = exportable_scales(amax)
    n_groups = cols // group_size
    if s0 is not None:
        s0 = jnp.asarray(s0, dtype=W.dtype)
        if s0.shape != (rows, n_groups):
            raise ValueError("s0 must have shape (M, N / group_size)")
    chosen = []
    work = W
    for group_index in range(n_groups):
        start = group_index * group_size
        block = work[:, start : start + group_size]
        block_factor = factor[start : start + group_size, start : start + group_size]
        metric = schur_from_factor(block_factor)
        if s0 is None:
            incoming = initial_group_scales(block, grid)
        else:
            incoming = s0[:, group_index]
        winner, _info = select_scales(
            block,
            block_factor,
            metric,
            incoming,
            grid,
            half_width=half_width,
            iterations=iterations,
            tie_tol=tie_tol,
        )
        work = commit_group(work, factor, start, winner, group_size)
        chosen.append(winner)
    return work, jnp.stack(chosen, axis=1)
