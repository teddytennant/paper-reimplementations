"""Structure-preserving block-wise magnitude–direction decomposition (§3.1, Appendix B.1)."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def _padded_width(n: int, block_size: int) -> tuple[int, int]:
    b = int(block_size)
    if b < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    k = (int(n) + b - 1) // b
    return k, k * b


def decompose(G: Array, block_size: int, eps: float = 1e-6) -> tuple[Array, Array]:
    """Decompose G into block magnitudes M and unit-direction V.

    G is (m, n). If B does not divide n, columns are right-zero-padded to
    npad, the smallest multiple of B that is >= n. Each padded row is split
    into k = npad / B contiguous 1 x B blocks g_{i,j}.

        m_{i,j} = ||g_{i,j}||_2
        (Vpad)_{i, jB:(j+1)B} = g_{i,j} / (m_{i,j} + eps)

    Padded columns are discarded so V has the original (m, n) shape.
    """
    G = jnp.asarray(G)
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D, got shape {G.shape}")
    m, n = int(G.shape[0]), int(G.shape[1])
    k, npad = _padded_width(n, block_size)
    Gpad = jnp.pad(G, ((0, 0), (0, npad - n)))
    blocks = Gpad.reshape(m, k, int(block_size))
    mag = jnp.linalg.norm(blocks, ord=2, axis=-1)
    v_blocks = blocks / (mag[..., None] + eps)
    v_pad = v_blocks.reshape(m, npad)
    return mag, v_pad[:, :n]


def repeat_blocks(M: Array, block_size: int, n: int) -> Array:
    """Rep_B(M): repeat each m_{i,j} across B columns, then crop to n."""
    M = jnp.asarray(M)
    if M.ndim != 2:
        raise ValueError(f"M must be 2-D, got shape {M.shape}")
    repeated = jnp.repeat(M, int(block_size), axis=1)
    return repeated[:, : int(n)]


def recompose(M: Array, V: Array, block_size: int, n: int | None = None) -> Array:
    """G ≈ Rep_B(M) ⊙ V, padding V if needed and cropping to n columns."""
    M = jnp.asarray(M)
    V = jnp.asarray(V)
    if n is None:
        n = int(V.shape[1])
    k, npad = _padded_width(n, block_size)
    if int(M.shape[1]) != k:
        k = int(M.shape[1])
        npad = k * int(block_size)
    v_pad = jnp.pad(V, ((0, 0), (0, max(0, npad - int(V.shape[1])))))[:, :npad]
    g_pad = repeat_blocks(M, block_size, npad) * v_pad
    return g_pad[:, : int(n)]
