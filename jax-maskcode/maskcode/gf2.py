"""GF(2) linear algebra, done in integer arithmetic.

Everything here stays in numpy uint8/int8 and uses XOR and AND. Nothing is a
float matmul: on an NVIDIA GPU a float32 matmul runs at TF32 by default, which
has a 10-bit mantissa, and a mod-2 product over a codeword longer than about
2048 bits would come back with the wrong parity. Integer arithmetic has no such
failure mode, and these matrices are tiny anyway.
"""

from __future__ import annotations

import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product over GF(2), computed exactly.

    Uses int64 accumulation followed by a mod-2 reduction. int64 holds the
    un-reduced sum exactly for any inner dimension we will ever use.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return (a @ b % 2).astype(np.uint8)


def rref(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2). Returns (rref, pivot columns)."""
    m = np.array(mat, dtype=np.uint8, copy=True)
    rows, cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(m[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            m[[r, i]] = m[[i, r]]
        others = np.nonzero(m[:, c])[0]
        others = others[others != r]
        if others.size:
            m[others] ^= m[r]
        pivots.append(c)
        r += 1
    return m, pivots


def nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis of the right null space of `mat` over GF(2), one vector per row.

    For a parity-check matrix H this returns a generator matrix G with
    G H^T = 0, which is the defining relation stated below (1) in the paper.
    """
    m = np.asarray(mat, dtype=np.uint8)
    _, cols = m.shape
    red, pivots = rref(m)
    free = [c for c in range(cols) if c not in pivots]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(pivots):
            basis[i, p] = red[r, f]
    return basis


def rank(mat: np.ndarray) -> int:
    return len(rref(mat)[1])


def systematic_generator(h: np.ndarray) -> np.ndarray:
    """Generator matrix for the code with parity-check matrix `h`."""
    g = nullspace(h)
    if g.size and matmul(g, h.T).any():
        raise ValueError("G H^T != 0; nullspace computation is wrong")
    return g


def all_messages(k: int) -> np.ndarray:
    """Every message in {0,1}^k, one per row. Only usable for small k."""
    idx = np.arange(2**k, dtype=np.uint64)[:, None]
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((idx >> shifts) & np.uint64(1)).astype(np.uint8)
