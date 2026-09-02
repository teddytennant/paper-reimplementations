"""FUSED-FIT, section 3.4: bounded two-pass construction of A_h and B_h.

The paper's contribution here is a fused CUDA kernel. This is the same bounded-panel
streaming structure written as a lax.scan over the calibration stream: pass one
reduces the weighted means of eq 9, pass two traverses observations in chunks of
size b, gathers the head-local blocks of eq 11 into contiguous per-head panels,
centers and weights them, and accumulates eqs 20-21 with head-batched matmuls. Per
chunk it holds O(Ht b (2 p_l + dt)) scratch, eq 22. It reproduces the ridge solution
and the memory bound, not the paper's 10.7x construction speedup.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .ridge import LAMBDA_DEFAULT, solve_from_stats


def _chunk_rows(n: int, chunk: int):
    n_chunks = -(-n // chunk)
    padded = n_chunks * chunk
    rows = jnp.concatenate([jnp.arange(n), jnp.zeros(padded - n, jnp.int32)])
    valid = jnp.concatenate([jnp.ones(n), jnp.zeros(padded - n)])
    return rows.reshape(n_chunks, chunk), valid.reshape(n_chunks, chunk)


@functools.partial(jax.jit, static_argnames=("chunk",))
def streaming_stats(src_z, tgt_z, layers, assign, w, chunk: int):
    """Stream eqs 20-21 in bounded panels. Returns (A, B, mu_x, mu_y).

    src_z is (Ls, n, Hs, ds) content-space source K or V, tgt_z is (n, Ht, dt),
    layers is S_l of length k, assign is a(h) of length Ht, w is (Ht, n) mean-one.
    """
    n, ht, dt = tgt_z.shape
    k = layers.shape[0]
    ds = src_z.shape[-1]
    p = k * ds
    rows_all, valid_all = _chunk_rows(n, chunk)
    dtype = tgt_z.dtype

    def panels(rows):
        """The gather-and-pack step: contiguous (Ht, b, k*ds) and (Ht, b, dt) panels."""
        blocks = src_z[layers[:, None, None], rows[None, :, None], assign[None, None, :], :]
        x = jnp.transpose(blocks, (2, 1, 0, 3)).reshape(ht, rows.shape[0], p)
        y = jnp.transpose(tgt_z[rows], (1, 0, 2))
        return x, y

    def mean_step(carry, xs):
        sx, sy = carry
        rows, valid = xs
        x, y = panels(rows)
        wc = w[:, rows] * valid[None, :]
        return (sx + jnp.einsum("hn,hnp->hp", wc, x),
                sy + jnp.einsum("hn,hnd->hd", wc, y)), None

    (sum_x, sum_y), _ = jax.lax.scan(
        mean_step,
        (jnp.zeros((ht, p), dtype), jnp.zeros((ht, dt), dtype)),
        (rows_all, valid_all),
    )
    mu_x, mu_y = sum_x / n, sum_y / n

    def stat_step(carry, xs):
        a_acc, b_acc = carry
        rows, valid = xs
        x, y = panels(rows)
        wc = w[:, rows] * valid[None, :]
        xc = x - mu_x[:, None, :]
        yc = y - mu_y[:, None, :]
        xw = xc * wc[:, :, None]
        return (a_acc + jnp.einsum("hnp,hnq->hpq", xw, xc),
                b_acc + jnp.einsum("hnp,hnd->hpd", xw, yc)), None

    (A, B), _ = jax.lax.scan(
        stat_step,
        (jnp.zeros((ht, p, p), dtype), jnp.zeros((ht, p, dt), dtype)),
        (rows_all, valid_all),
    )
    return A, B, mu_x, mu_y


def fused_fit(src_z, tgt_z, layers, assign, w, lam: float = LAMBDA_DEFAULT, chunk: int = 256):
    """Streaming construction followed by the eq 23 solve. Returns (W, b)."""
    A, B, mu_x, mu_y = streaming_stats(src_z, tgt_z, layers, assign, w, chunk)
    return solve_from_stats(A, B, mu_x, mu_y, lam)


def scratch_elements(n_tgt_heads: int, chunk: int, p_local: int, dt: int) -> int:
    """The eq 22 bound O(Ht b (2 p_l + dt)) as an element count."""
    return n_tgt_heads * chunk * (2 * p_local + dt)
