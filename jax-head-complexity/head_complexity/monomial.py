"""Theorem 5, the monomial vote: one additive head per monomial of f.

Every f: {0,1}^n -> {0,1} agrees on the cube with its multilinear (Mobius)
expansion f(x) = sum_S alpha_S prod_{i in S} x_i, and those monomials are
monotone terms. Lemma 2 builds one additive head whose scalar read-out A_S(x)
sits within n e^{-R/2} of prod_{i in S} x_i, using the query token as its own
reference so no token is added to the input. Theorem 5 reads the heads out in
the affine scalar form of Definition 2 with head weights alpha_S and threshold
theta = 1/2 - alpha_empty, giving at most 2^n heads.
"""

from __future__ import annotations

import itertools
import math

import jax.numpy as jnp
import numpy as np

from head_complexity.layer import AdditiveEmbedding, AffineLayer, AttentionCore


def subsets(n: int) -> list[tuple[int, ...]]:
    """All 2^n subsets of [n], ordered by the integer whose bits they are."""
    out = []
    for mask in range(1 << n):
        out.append(tuple(i for i in range(n) if mask >> i & 1))
    return out


def mobius_coefficients(truth_table: np.ndarray) -> dict[tuple[int, ...], int]:
    """alpha_S = sum_{T subseteq S} (-1)^{|S| - |T|} f(1_T), Section 5.1.

    Index convention throughout this module: bit i of the table index is x_{i+1},
    least significant bit first, and a subset S is the sorted tuple of those
    0-based positions. `inputs_in_mobius_order` builds the matching input rows.
    The coefficients are integers with |alpha_S| <= 2^{|S|}.
    """
    f = np.asarray(truth_table, dtype=np.int64)
    n = int(round(math.log2(f.size)))
    if f.size != 1 << n:
        raise ValueError("truth table length must be a power of two")
    alpha: dict[tuple[int, ...], int] = {}
    for S in subsets(n):
        total = 0
        for r in range(len(S) + 1):
            for T in itertools.combinations(S, r):
                idx = 0
                for i in T:
                    idx |= 1 << i
                total += (-1) ** (len(S) - r) * int(f[idx])
        alpha[S] = total
    return alpha


def truth_table_from_mobius(alpha: dict[tuple[int, ...], int], n: int) -> np.ndarray:
    """Evaluate sum_S alpha_S prod_{i in S} x_i back on the whole cube."""
    out = np.zeros(1 << n, dtype=np.int64)
    for idx in range(1 << n):
        x = [(idx >> i) & 1 for i in range(n)]
        total = 0
        for S, a in alpha.items():
            if a == 0:
                continue
            if all(x[i] for i in S):
                total += a
        out[idx] = total
    return out


def truth_table_of(f, n: int) -> np.ndarray:
    """Tabulate a python predicate on bit tuples, in the index order above."""
    out = np.zeros(1 << n, dtype=np.int64)
    for idx in range(1 << n):
        out[idx] = int(f(tuple((idx >> i) & 1 for i in range(n))))
    return out


def inputs_in_mobius_order(n: int) -> np.ndarray:
    """The rows of {0,1}^n ordered to match the Mobius index (bit i is x_{i+1})."""
    return np.array(
        [[(idx >> i) & 1 for i in range(n)] for idx in range(1 << n)], dtype=np.int32
    )


def steepness(n: int, slack: float = 8.0) -> float:
    """R from the proof of Theorem 5.

    The error is bounded by n e^{-R/2} sum_S |alpha_S| <= n 3^n e^{-R/2}, so any
    R > 2(n ln 3 + ln 2n) puts it under 1/2. `slack` buys headroom without
    pushing e^{-3R} anywhere near the float64 floor.
    """
    return 2.0 * (n * math.log(3.0) + math.log(2.0 * n)) + slack


def monotone_term_head(S, n: int, R: float, weight: float = 1.0):
    """Lemma 2, as (xi, value_column) in the coordinate frame of `monomial_vote_layer`.

    The head uses the query as its reference: r_? = e^{-R/2}, the query's value
    reads 1 (scaled here by `weight`), every bit token's value reads 0, and the
    exponentiated bit scores are

        r_{x_i,i} = e^{-R x_i}        (i in S),
        r_{x_i,i} = e^{-2R - R x_i}   (i not in S).

    That is additive: the bit slope xi . (tau_1 - tau_0) = -R is one constant per
    head, and the position biases xi . pi_i are 0 on S and -2R off S. The scalar
    read-out is then the query's own attention weight,

        A_S(x) = e^{-R/2} / (e^{-R/2} + sum_{i in S} e^{-R x_i} + sum_{i not in S} e^{-2R - R x_i}),

    which is 1 - O(n e^{-R/2}) when every bit of S is on and at most e^{-R/2}
    when one is off, because a zero bit in S contributes e^0 = 1.
    """
    d = n + 3
    xi = np.zeros(d)
    xi[0] = 0.0  # score of token 0 at a position in S
    xi[1] = -R  # score of token 1: the bit slope
    for i in range(n):
        xi[2 + i] = 0.0 if i in S else -2.0 * R
    xi[n + 2] = -R / 2.0  # the query's self-score
    value = np.zeros(d)
    value[n + 2] = weight  # only the query carries a value
    return xi, value


def monomial_vote_layer(truth_table: np.ndarray, R: float | None = None) -> AffineLayer:
    """Theorem 5. Build the affine-scalar layer computing `truth_table`.

    One Lemma 2 head per nonempty S with alpha_S != 0, all sharing a common R,
    with head weights alpha_S folded into the value matrices (d_out = 1, omega = 1)
    and threshold theta = 1/2 - alpha_empty. If f has no nonempty monomial, a
    single zero-valued head is used, so the head count is between 1 and 2^n.

    The embedding is additive with d = n + 3: tau_0 = e_1, tau_1 = e_2,
    pi_i = e_{i+2}, Phi(?) = e_{n+3}. The value matrices touch only coordinate
    n+3, which the input tokens do not carry, so their read-out is 0 as Lemma 2
    requires.
    """
    f = np.asarray(truth_table, dtype=np.int64)
    n = int(round(math.log2(f.size)))
    if R is None:
        R = steepness(n)
    alpha = mobius_coefficients(f)

    terms = [(S, a) for S, a in alpha.items() if S and a != 0]
    if not terms:
        terms = [((), 0)]  # one zero-valued head, as in the proof

    d = n + 3
    A = np.zeros((len(terms), d, d))
    V = np.zeros((len(terms), d, 1))
    for h, (S, a) in enumerate(terms):
        xi, value = monotone_term_head(S, n, R, weight=float(a))
        A[h, n + 2, :] = xi  # v_? A_h = xi_h
        V[h, :, 0] = value

    tau = np.zeros((2, d))
    tau[0, 0] = 1.0
    tau[1, 1] = 1.0
    pi = np.zeros((n, d))
    for i in range(n):
        pi[i, 2 + i] = 1.0
    v_query = np.zeros(d)
    v_query[n + 2] = 1.0

    emb = AdditiveEmbedding(
        jnp.asarray(tau, dtype=jnp.float64),
        jnp.asarray(pi, dtype=jnp.float64),
        jnp.asarray(v_query, dtype=jnp.float64),
    )
    core = AttentionCore(emb, jnp.asarray(A, dtype=jnp.float64), jnp.asarray(V, dtype=jnp.float64))
    theta = 0.5 - float(alpha[()])
    return AffineLayer(core, jnp.ones((1,), dtype=jnp.float64), theta)


def head_count(truth_table: np.ndarray) -> int:
    """The number of heads Theorem 5 spends: one per nonzero nonempty monomial."""
    alpha = mobius_coefficients(truth_table)
    return max(1, sum(1 for S, a in alpha.items() if S and a != 0))


def error_bound(truth_table: np.ndarray, R: float | None = None) -> float:
    """n e^{-R/2} sum_S |alpha_S|, the slack the proof of Theorem 5 leaves.

    The decision succeeds when this is below 1/2.
    """
    f = np.asarray(truth_table, dtype=np.int64)
    n = int(round(math.log2(f.size)))
    if R is None:
        R = steepness(n)
    total = sum(abs(a) for a in mobius_coefficients(f).values())
    return n * math.exp(-R / 2.0) * total


def evaluate(layer: AffineLayer, n: int) -> np.ndarray:
    """Run the layer on {0,1}^n in Mobius index order."""
    return np.asarray(layer.decide(jnp.asarray(inputs_in_mobius_order(n))))


__all__ = [
    "error_bound",
    "evaluate",
    "head_count",
    "inputs_in_mobius_order",
    "mobius_coefficients",
    "monomial_vote_layer",
    "monotone_term_head",
    "steepness",
    "subsets",
    "truth_table_from_mobius",
    "truth_table_of",
]
