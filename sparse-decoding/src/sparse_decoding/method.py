"""SparseDecoding equations, as a CPU reference.

Calibration columns are decode-step activations only (Eq. 4). The layer
objective is the squared residual on those columns (Eqs. 1 and 5). OBS uses
H = 2 X X^T, saliency L_q = w_q^2 / (2 [H^{-1}]_qq), and the compensation
delta w = -w_q / [H^{-1}]_qq * H^{-1}_{:,q} (Eq. 2). N:M keeps the n largest
saliencies in each group of m. The SpMV stores one bitmask bit per column and
recovers the next nonzero with a least-significant-set-bit scan.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def build_x_ar(activations, decode_mask):
    """Stack decode-step activations as columns of X_AR.

    activations: (d_in, T) or a list of (d_in, T_m) arrays, one per prompt.
    decode_mask: a boolean mask of length T, or a list of per-prompt masks or
    prefill lengths. Prefill positions are dropped. N_D is the number of
    retained columns (Eq. 4).
    """
    if isinstance(activations, (list, tuple)):
        columns = []
        for act, mask in zip(activations, decode_mask, strict=True):
            block = build_x_ar(act, mask)
            if block.shape[1]:
                columns.append(block)
        if not columns:
            d_in = int(np.asarray(activations[0]).shape[0])
            return jnp.zeros((d_in, 0), dtype=jnp.asarray(activations[0]).dtype)
        return jnp.concatenate(columns, axis=1)

    act = jnp.asarray(activations)
    if act.ndim != 2:
        raise ValueError("activations must have shape (d_in, T)")
    mask = _decode_mask(decode_mask, act.shape[1])
    return act[:, mask]


def _decode_mask(decode_mask, length):
    if isinstance(decode_mask, (int, np.integer)):
        n_prefill = int(decode_mask)
        if n_prefill < 0 or n_prefill > length:
            raise ValueError("prefill length out of range")
        return np.arange(length) >= n_prefill
    mask = np.asarray(decode_mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError("decode_mask length must match the activation width")
    return mask


def reconstruction_loss(weight, sparse_weight, activations):
    """||(W - W_hat) X||_F^2. X has activations as columns."""
    residual = jnp.asarray(weight) - jnp.asarray(sparse_weight)
    acted = residual @ jnp.asarray(activations)
    return jnp.sum(acted * acted)


def gram_hessian(activations):
    """H = 2 X X^T, the OBS Hessian in Eq. 2. X columns are activations."""
    act = jnp.asarray(activations)
    return 2.0 * (act @ act.T)


def population_hessian(activations):
    """H = 2 E[x x^T], estimated by the column mean (Eq. 6)."""
    act = jnp.asarray(activations)
    n = act.shape[1]
    if n == 0:
        raise ValueError("population Hessian needs at least one column")
    return 2.0 * (act @ act.T) / n


def obs_saliency_and_update(row, h_inv, index):
    """Eq. 2 for one weight. Returns L_q and the full compensation delta."""
    row = jnp.asarray(row)
    h_inv = jnp.asarray(h_inv)
    denom = h_inv[index, index]
    saliency = (row[index] ** 2) / (2.0 * denom)
    delta = -(row[index] / denom) * h_inv[:, index]
    return saliency, delta


def downdate_inverse(h_inv, index):
    """Rank-one OBS downdate after removing column `index`."""
    h_inv = jnp.asarray(h_inv)
    column = h_inv[:, index]
    return h_inv - jnp.outer(column, column) / h_inv[index, index]


def apply_obs_removal(row, h_inv, index, freeze_prefix=True):
    """Remove weight `index` and compensate.

    freeze_prefix=True matches SparseGPT's fixed left-to-right order: columns
    at or before `index` stay put, and the Eq. 2 update is applied only to
    later columns. freeze_prefix=False applies the full Eq. 2 update.
    """
    saliency, delta = obs_saliency_and_update(row, h_inv, index)
    row = jnp.asarray(row)
    if freeze_prefix:
        updated = row.at[index + 1 :].add(delta[index + 1 :])
        updated = updated.at[index].set(0.0)
    else:
        updated = (row + delta).at[index].set(0.0)
    return updated, downdate_inverse(h_inv, index), saliency


def _keep_in_group(row, h_inv, start, n, m):
    """Keep the n largest Eq. 2 saliencies in one group. Ties take the lower index."""
    if m <= 0 or n <= 0 or n > m:
        raise ValueError("need 0 < n <= m")
    diag = np.asarray(jnp.diag(h_inv), dtype=np.float64)[start : start + m]
    weights = np.asarray(row, dtype=np.float64)[start : start + m]
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("inverse Hessian diagonal is not positive in this group")
    saliency = (weights ** 2) / (2.0 * diag)
    order = np.lexsort((np.arange(m), -saliency))
    keep = np.zeros(m, dtype=bool)
    keep[order[:n]] = True
    return keep


def nm_prune(weight, activations, n, m, damp=0.0):
    """N:M OBS prune. Returns sparse weights and the boolean keep-mask.

    Saliency inside a group uses the inverse Hessian at the start of that
    group. Pruned columns are then removed left to right, with the Eq. 2
    update applied only to later columns, and the inverse Hessian downdated.
    `damp` is added as damp * I before the inversion. It is a numerical
    stabilizer, not a published constant. The default is 0 so Eq. 2 is exact
    when H is nonsingular.
    """
    weight = jnp.asarray(weight)
    if weight.ndim != 2:
        raise ValueError("weight must have shape (d_out, d_in)")
    act = jnp.asarray(activations, dtype=weight.dtype)
    hessian = gram_hessian(act)
    eye = jnp.eye(hessian.shape[0], dtype=hessian.dtype)
    hessian = hessian + damp * eye
    base_inv = jnp.linalg.inv(hessian)
    sparse = np.array(weight, dtype=np.float64)
    base = np.array(base_inv, dtype=np.float64)
    mask = np.zeros(sparse.shape, dtype=bool)
    for r in range(sparse.shape[0]):
        h_inv = jnp.asarray(base)
        row = jnp.asarray(sparse[r])
        d_in = int(row.shape[0])
        if d_in % m != 0:
            raise ValueError("d_in must be a multiple of m")
        for start in range(0, d_in, m):
            keep = _keep_in_group(row, h_inv, start, n, m)
            mask[r, start : start + m] = keep
            for offset, kept in enumerate(np.asarray(keep)):
                if kept:
                    continue
                row, h_inv, _ = apply_obs_removal(
                    row, h_inv, start + offset, freeze_prefix=True
                )
        sparse[r] = np.asarray(row)
        sparse[r, ~mask[r]] = 0.0
    out_dtype = weight.dtype
    return {
        "weight": jnp.asarray(sparse, dtype=out_dtype),
        "mask": jnp.asarray(mask),
    }


def pack_bitmask(mask):
    """One bit per column, packed into uint32 words. Bit 0 is column 0."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape (rows, cols)")
    rows, cols = mask.shape
    n_words = (cols + 31) // 32
    words = np.zeros((rows, n_words), dtype=np.uint32)
    for col in range(cols):
        bit = np.uint32(1) << np.uint32(col % 32)
        words[:, col // 32] |= np.uint32(bit) * mask[:, col].astype(np.uint32)
    return words


def pack_values(weight, mask):
    """Kept weights in increasing column order. nnz must be constant per row."""
    weight = np.asarray(weight)
    mask = np.asarray(mask, dtype=bool)
    counts = mask.sum(axis=1)
    if np.any(counts != counts[0]):
        raise ValueError("every row must keep the same number of weights")
    nnz = int(counts[0])
    values = np.zeros((weight.shape[0], nnz), dtype=np.float64)
    for r in range(weight.shape[0]):
        values[r] = weight[r, mask[r]]
    return values


def bits_per_nonzero(mask):
    """Stored bitmask bits divided by the number of kept weights.

    At 50% sparsity with a column count divisible by 32 this is 2, versus 32
    for an int32 column index.
    """
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    n_words = (cols + 31) // 32
    nnz = int(mask.sum())
    if nnz == 0:
        raise ValueError("mask has no nonzeros")
    return (rows * n_words * 32) / nnz


def _ctz(bit):
    return int(bit).bit_length() - 1


def bitmask_spmv(values, bitmask, vector, steps=None):
    """N:M SpMV. Column indices come only from a fixed-step bit scan.

    For each uint32 word, clear the least significant set bit `steps` times
    (or popcount(word) times if steps is None). At 50% N:M with M <= 32 and
    a column count divisible by 32, every word has exactly 16 set bits, so
    steps=16 is the compile-time loop bound from the paper.
    """
    values = np.asarray(values, dtype=np.float64)
    bitmask = np.asarray(bitmask, dtype=np.uint32)
    vector = np.asarray(vector, dtype=np.float64)
    rows, n_words = bitmask.shape
    out = np.zeros(rows, dtype=np.float64)
    for r in range(rows):
        cursor = 0
        acc = 0.0
        for word_index in range(n_words):
            bits = int(bitmask[r, word_index])
            n_steps = int(bits.bit_count()) if steps is None else int(steps)
            if steps is not None and int(bits.bit_count()) != n_steps:
                raise ValueError(
                    f"word {word_index} of row {r} has {bits.bit_count()} set bits, expected {n_steps}"
                )
            for _ in range(n_steps):
                lowest = bits & -bits
                if lowest == 0:
                    raise ValueError("bit scan ran out of set bits")
                col = word_index * 32 + _ctz(lowest)
                acc += values[r, cursor] * vector[col]
                cursor += 1
                bits &= bits - 1
        out[r] = acc
    return jnp.asarray(out)


def calibration_discrepancy(h_dec, h_cal, k):
    """Normalized decode-time calibration discrepancy, Eqs. 9-10.

    eps = ||Lambda^{-1/2} U^T H_cal U Lambda^{-1/2} - I||_op, where U and
    Lambda are the top-k eigenpairs of H_dec. eps is 0 when the calibration
    Hessian matches the decode Hessian on that subspace.
    """
    h_dec = jnp.asarray(h_dec)
    h_cal = jnp.asarray(h_cal)
    evals, evecs = jnp.linalg.eigh(h_dec)
    order = jnp.argsort(evals)[::-1]
    chosen = order[:k]
    lam = evals[chosen]
    if jnp.any(lam <= 0):
        raise ValueError("top-k decode eigenvalues must be positive")
    basis = evecs[:, chosen]
    scale = 1.0 / jnp.sqrt(lam)
    restricted = basis.T @ h_cal @ basis
    normalized = restricted * scale[:, None] * scale[None, :]
    return jnp.linalg.norm(normalized - jnp.eye(k, dtype=normalized.dtype), ord=2)


def projected_reconstruction_loss(delta_weight, hessian, basis):
    """Eq. 12: 1/2 tr(dW P H P dW^T), P = U U^T."""
    delta_weight = jnp.asarray(delta_weight)
    hessian = jnp.asarray(hessian)
    basis = jnp.asarray(basis)
    projector = basis @ basis.T
    middle = projector @ hessian @ projector
    return 0.5 * jnp.trace(delta_weight @ middle @ delta_weight.T)
