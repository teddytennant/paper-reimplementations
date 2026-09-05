"""Outer linear block codes: G, H, syndromes, Tanner graph, attention mask.

Section II-A of the paper. An (n, k) binary linear code is given by a generator
G in {0,1}^(k x n) and a parity check H in {0,1}^((n-k) x n) with G H^T = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import gf2
from .ops import masked_prod, stable_arctanh

# ---------------------------------------------------------------------------
# GF(2^m) helpers, used only to build BCH generator polynomials
# ---------------------------------------------------------------------------

_PRIMITIVE = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011}


def _gf_tables(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Exponential and log tables for GF(2^m) with the primitive poly above."""
    poly = _PRIMITIVE[m]
    size = 1 << m
    order = size - 1
    exp = np.zeros(2 * order, dtype=np.int64)
    log = np.zeros(size, dtype=np.int64)
    x = 1
    for i in range(order):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & size:
            x ^= poly
    exp[order:] = exp[:order]
    return exp, log


def _gf_mul(a: int, b: int, exp: np.ndarray, log: np.ndarray) -> int:
    if a == 0 or b == 0:
        return 0
    return int(exp[log[a] + log[b]])


def _poly_mul_gf(p: list[int], q: list[int], exp, log) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            out[i + j] ^= _gf_mul(a, b, exp, log)
    return out


def _minimal_polys(m: int, t: int) -> list[int]:
    """Binary generator polynomial of the narrow-sense BCH code, low bit first.

    The generator is the least common multiple of the minimal polynomials of
    alpha^1 .. alpha^(2t), one per binary cyclotomic coset.
    """
    exp, log = _gf_tables(m)
    order = (1 << m) - 1
    seen: set[int] = set()
    gen = [1]
    for i in range(1, 2 * t + 1):
        if i in seen:
            continue
        coset = []
        j = i
        while j not in coset:
            coset.append(j)
            j = (2 * j) % order
        seen.update(coset)
        minpoly = [1]
        for c in coset:
            minpoly = _poly_mul_gf(minpoly, [int(exp[c]), 1], exp, log)
        assert all(v in (0, 1) for v in minpoly), "minimal poly left GF(2)"
        gen = _poly_mul_gf(gen, minpoly, exp, log)
    return gen


# ---------------------------------------------------------------------------
# The code object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearBlockCode:
    """An (n, k) binary linear block code plus everything MaskCode needs from it.

    `h` and `g` are numpy uint8 and stay that way. Every GF(2) operation runs in
    integer arithmetic (see maskcode.gf2), never as a float matmul, so nothing
    here can be corrupted by TF32 on a GPU.
    """

    name: str
    h: np.ndarray
    g: np.ndarray

    def __post_init__(self) -> None:
        if gf2.matmul(self.g, self.h.T).any():
            raise ValueError(f"{self.name}: G H^T != 0 over GF(2)")

    @property
    def n(self) -> int:
        return int(self.h.shape[1])

    @property
    def k(self) -> int:
        return int(self.g.shape[0])

    @property
    def n_checks(self) -> int:
        """Rows of H. Equals n - k only when H has full row rank."""
        return int(self.h.shape[0])

    @property
    def rate(self) -> float:
        return self.k / self.n

    @property
    def check_degrees(self) -> np.ndarray:
        """|N_v(i)| for each check node, the row weights of H."""
        return self.h.sum(axis=1).astype(np.int64)

    # -- encoding and syndromes --------------------------------------------

    def encode(self, m: np.ndarray) -> np.ndarray:
        """Equation (1): c = m G over GF(2). `m` is (..., k), result (..., n)."""
        m = np.asarray(m, dtype=np.uint8)
        flat = m.reshape(-1, self.k)
        return gf2.matmul(flat, self.g).reshape(*m.shape[:-1], self.n)

    def hard_syndrome(self, u: np.ndarray) -> np.ndarray:
        """u H^T over GF(2). Zero iff u is a codeword (Sec. II-A)."""
        u = np.asarray(u, dtype=np.uint8)
        flat = u.reshape(-1, self.n)
        out = gf2.matmul(flat, self.h.T)
        return out.reshape(*u.shape[:-1], self.n_checks)

    def codewords(self) -> np.ndarray:
        """Every codeword. Only call this for small k."""
        return self.encode(gf2.all_messages(self.k))

    # -- Tanner graph and attention mask -----------------------------------

    def tanner_adjacency(self) -> np.ndarray:
        """Equation (23): A = A_intra + A_inter, shape (2n-k, 2n-k).

            A = [[ 1(H^T H > 0),  H^T ],
                 [       H,       I   ]]

        The top-left block is the Gram matrix indicator: two variable nodes are
        connected when they share at least one check node. The off-diagonal
        blocks are the bipartite Tanner graph itself, so variable node v attends
        to check node c exactly when H[c, v] == 1. The bottom-right block is the
        identity, so check nodes see only themselves.

        Built with integer matmul, not float, for the reason in maskcode.gf2.
        """
        h = np.asarray(self.h, dtype=np.int64)
        n, mrows = self.n, self.n_checks
        gram = (h.T @ h > 0).astype(np.uint8)
        top = np.concatenate([gram, h.T.astype(np.uint8)], axis=1)
        bottom = np.concatenate(
            [h.astype(np.uint8), np.eye(mrows, dtype=np.uint8)], axis=1
        )
        return np.concatenate([top, bottom], axis=0)

    def attention_mask(self) -> jax.Array:
        """Equation (24): M = log_2(A), i.e. 0 where A == 1 and -inf where A == 0.

        Added to the attention logits just before the softmax, as in (8).
        """
        a = self.tanner_adjacency()
        return jnp.where(jnp.asarray(a) == 1, 0.0, -jnp.inf)

    def zero_mask(self) -> jax.Array:
        """The all-zero mask O_n used by the initial-phase encoder in (15)."""
        return jnp.zeros((self.n, self.n))

    # -- soft syndrome ------------------------------------------------------

    def soft_syndrome(self, r: jax.Array, eps: float | None = None) -> jax.Array:
        """Equation (19), implemented sign for sign.

            s_i = 2 arctanh( prod_{b in N_v(i)} tanh(r_b / 2) ),   i in [n-k]

        where N_v(i) = { b : H[i, b] == 1 } and r is the per-bit LLR vector from
        (18), in the paper's convention r_b = log(Pr(c_b=1)/Pr(c_b=0)).

        Sign of the result. Let d_i be the degree of check i and syn_i the hard
        syndrome bit. As |r| grows, tanh(r_b/2) -> +1 when the hard decision is
        1 and -1 when it is 0, so

            sign(s_i) -> (-1)^(d_i + syn_i).

        This is the standard sum-product check node update multiplied by
        (-1)^d_i, and the factor is there because the paper defines its LLR as
        log(p1/p0) rather than the usual log(p0/p1). The paper's reading, "s_i
        << 0 indicates a probable violation of the i-th parity constraint",
        therefore holds exactly on checks of even degree, and is inverted on
        checks of odd degree. Every H built in this module happens to have only
        even row weights, because each of these codes contains the all-ones
        vector and so every dual codeword is orthogonal to it. The odd case is
        still handled correctly and is covered by a test on a hand-made H.

        `r` may carry leading batch axes; the last axis must be n.
        """
        mask = jnp.asarray(self.h, dtype=bool)
        t = jnp.tanh(r / 2.0)
        # (..., 1, n) against (n_checks, n) broadcasts to (..., n_checks, n)
        prod = masked_prod(t[..., None, :], mask, axis=-1)
        if eps is None:
            return 2.0 * stable_arctanh(prod)
        return 2.0 * stable_arctanh(prod, eps)


# ---------------------------------------------------------------------------
# Concrete codes
# ---------------------------------------------------------------------------


def hamming74() -> LinearBlockCode:
    """The (7,4) Hamming code, a perfect single-error-correcting code.

    All three rows of H have weight 4, so the even-degree branch of the soft
    syndrome sign rule applies.
    """
    h = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return LinearBlockCode("Hamming(7,4)", h, gf2.systematic_generator(h))


def single_parity_check(n: int = 4) -> LinearBlockCode:
    """The (n, n-1) single parity check code.

    Its Tanner graph is a star with one check node and therefore has no cycles,
    which makes sum-product BP exactly optimal on it. That is what the exact
    BP-equals-ML test uses.
    """
    h = np.ones((1, n), dtype=np.uint8)
    return LinearBlockCode(f"SPC({n},{n-1})", h, gf2.systematic_generator(h))


def bch(m: int = 4, t: int = 2) -> LinearBlockCode:
    """Narrow-sense binary BCH code of length n = 2^m - 1 correcting t errors.

    m=4, t=2 gives BCH(15,7), the small stand-in for the paper's BCH(31,16).
    G is the cyclic shift matrix of the generator polynomial and H is a basis of
    its null space, so G H^T = 0 holds by construction.
    """
    n = (1 << m) - 1
    gen = _minimal_polys(m, t)
    deg = len(gen) - 1
    k = n - deg
    g = np.zeros((k, n), dtype=np.uint8)
    for i in range(k):
        g[i, i : i + deg + 1] = np.array(gen, dtype=np.uint8)
    h = gf2.nullspace(g)
    return LinearBlockCode(f"BCH({n},{k})", h, g)


def array_ldpc(ell: int = 7, dv: int = 2, dc: int = 4) -> LinearBlockCode:
    """Regular array LDPC: H is a dv x dc grid of circulant shifts of I_ell.

    Block (i, j) is I_ell cyclically shifted by (i*j mod ell). With ell prime and
    dc <= ell the girth is at least 6, so there are no 4-cycles. Every column has
    weight dv and every row weight dc, so all check degrees are even when dc is.
    Default gives a (28, 15) code, the small stand-in for LDPC(49,24).
    """
    blocks = []
    eye = np.eye(ell, dtype=np.uint8)
    for i in range(dv):
        row = [np.roll(eye, (i * j) % ell, axis=1) for j in range(dc)]
        blocks.append(np.concatenate(row, axis=1))
    h = np.concatenate(blocks, axis=0)
    g = gf2.systematic_generator(h)
    n = ell * dc
    return LinearBlockCode(f"ArrayLDPC({n},{g.shape[0]})", h, g)


def tree_code() -> LinearBlockCode:
    """A (7,4) code whose Tanner graph is a tree, so BP on it is exact.

    Three weight-3 checks chained through variables 2 and 4:

        H = [[1,1,1,0,0,0,0],
             [0,0,1,1,1,0,0],
             [0,0,0,0,1,1,1]]

    Two uses. Sum-product on a cycle-free graph returns the exact bitwise
    posterior, which is what pins the BP implementation down against exhaustive
    enumeration. And the rows have odd weight, unlike every other code here, so
    it also exercises the (-1)^d_i branch of the soft syndrome sign rule.
    """
    h = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return LinearBlockCode("Tree(7,4)", h, gf2.systematic_generator(h))
