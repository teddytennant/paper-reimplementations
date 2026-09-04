"""Theorem 2: k heads compute k-bit parity, in embedding dimension d = k + 3.

The construction (Section A.2) runs the rational-function view of Section 3.1
backwards. On an input of Hamming weight c, head j is arranged to have

    denominator  d_x^j = c + j,      numerator gap  s_x^j = alpha_j + beta_j c,

so the output-0 gap is the rational function Delta(c) = sum_j (alpha_j + beta_j c)/(c + j).
Lemma 4 supplies coefficients making that sum equal P(c)/D(c) with

    D(t) = prod_{j=1..k} (t + j),   P(t) = (-1)^k prod_{m=0..k-1} (t - m - 1/2),

whose sign on the integers c = 0, ..., k is exactly (-1)^c. Since out(x) = 1 iff
Delta(x) < 0, that is parity.
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from head_complexity.layer import (
    AdditiveEmbedding,
    ArgmaxLayer,
    AttentionCore,
    all_inputs,
)

DEFAULT_MASK = 40.0
"""The finite stand-in for M -> infinity in equation (3). The query slot then
contributes e^{-M} to each denominator and, because u_?^j = 0, nothing to any
numerator, so Delta(c) = sum_j (alpha_j + beta_j c)/(c + j + e^{-M}). At M = 40
that perturbation is 4e-18, far below the smallest |P(c)/D(c)| for k <= 8."""


def D_poly(t, k: int):
    """D(t) = prod_{j=1..k} (t + j), from Lemma 4. Exact on Fraction input."""
    out = Fraction(1) if isinstance(t, Fraction) else 1.0
    for j in range(1, k + 1):
        out = out * (t + j)
    return out


def P_poly(t, k: int):
    """P(t) = (-1)^k prod_{m=0..k-1} (t - m - 1/2), from Lemma 4."""
    half = Fraction(1, 2) if isinstance(t, Fraction) else 0.5
    out = Fraction(-1) ** k if isinstance(t, Fraction) else float((-1) ** k)
    for m in range(k):
        out = out * (t - m - half)
    return out


def partial_fractions(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Lemma 4: alpha_j, beta_j with P(t)/D(t) = sum_j (alpha_j + beta_j t)/(t + j).

    deg P = deg D and the roots -1, ..., -k of D are simple, so
    P/D = C + sum_j lambda_j/(t + j) with C = (-1)^k the ratio of leading
    coefficients and lambda_j = P(-j)/D'(-j). The constant is spread across the
    k terms as C = sum_j (C/k)(t + j)/(t + j), giving

        alpha_j = lambda_j + C j / k,    beta_j = C / k.

    Residues are computed over Fraction, so the coefficients are exact before
    they are rounded once into float64.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    C = Fraction((-1) ** k)
    alpha, beta = [], []
    for j in range(1, k + 1):
        num = P_poly(Fraction(-j), k)
        den = Fraction(1)
        for m in range(1, k + 1):  # D'(-j) = prod_{m != j} (m - j)
            if m != j:
                den *= m - j
        lam = num / den
        alpha.append(lam + C * Fraction(j, k))
        beta.append(C / k)
    return (
        np.array([float(a) for a in alpha], dtype=np.float64),
        np.array([float(b) for b in beta], dtype=np.float64),
    )


def gap_at_weight(k: int, c: int) -> float:
    """The idealised Delta(c) = P(c)/D(c) of Theorem 2, at M -> infinity."""
    return float(P_poly(Fraction(c), k) / D_poly(Fraction(c), k))


def construction_margin(k: int) -> float:
    """min_c |P(c)/D(c)| over c = 0, ..., k.

    This is the margin the construction actually achieves, and the number
    tolerances in the tests are set against.
    """
    return min(abs(gap_at_weight(k, c)) for c in range(k + 1))


def parity_layer(k: int, n: int | None = None, mask: float = DEFAULT_MASK) -> ArgmaxLayer:
    """The k-head arg-max layer of Theorem 2, on n input positions (default n = k).

    Embedding (additive, d = n + 3): Phi(0, i) = e_1 + e_{i+2},
    Phi(1, i) = e_2 + e_{i+2}, Phi(?) = e_{n+3}. Effective queries, equation (3):

        xi_j = log(j/k) e_1 + log(1 + j/k) e_2 - M e_{n+3},

    so every position scores j/k on token 0 and 1 + j/k on token 1 and the
    denominator collapses to (k - c)(j/k) + c(1 + j/k) = c + j when n = k.
    Values: d_out = 2, W_O = I_2, V_j e_1 = delta_j/2 and V_j e_2 = -delta_j/2 so
    that V_j(w_0 - w_1)^T = delta_j = eta_{0,j} e_1 + eta_{1,j} e_2 with

        eta_{0,j} = alpha_j / j,   eta_{1,j} = (beta_j + alpha_j/k) / (1 + j/k).

    Passing n > k gives the same k heads on more input bits. That is not a
    parity solver, it is the object Theorem 1 rules out, and `obstruction.py`
    uses it to watch the alternating sum vanish.
    """
    if n is None:
        n = k
    d = n + 3
    q = n + 2  # index of the query coordinate e_{n+3}

    tau = np.zeros((2, d))
    tau[0, 0] = 1.0
    tau[1, 1] = 1.0
    pi = np.zeros((n, d))
    for i in range(n):
        pi[i, i + 2] = 1.0
    v_query = np.zeros(d)
    v_query[q] = 1.0

    alpha, beta = partial_fractions(k)
    A = np.zeros((k, d, d))
    V = np.zeros((k, d, 2))
    for idx in range(k):
        j = idx + 1
        xi = np.zeros(d)
        xi[0] = np.log(j / k)
        xi[1] = np.log1p(j / k)
        xi[q] = -mask
        A[idx, q, :] = xi  # v_? A_j = xi_j, since v_? is the q-th basis vector

        eta0 = alpha[idx] / j
        eta1 = (beta[idx] + alpha[idx] / k) / (1.0 + j / k)
        delta = np.zeros(d)
        delta[0] = eta0
        delta[1] = eta1
        V[idx, :, 0] = delta / 2.0
        V[idx, :, 1] = -delta / 2.0

    emb = AdditiveEmbedding(
        jnp.asarray(tau, dtype=jnp.float64),
        jnp.asarray(pi, dtype=jnp.float64),
        jnp.asarray(v_query, dtype=jnp.float64),
    )
    core = AttentionCore(emb, jnp.asarray(A, dtype=jnp.float64), jnp.asarray(V, dtype=jnp.float64))
    return ArgmaxLayer(core, jnp.eye(2, dtype=jnp.float64))


def parity(xs) -> np.ndarray:
    """The target bit, XOR of the input bits."""
    return np.asarray(xs).sum(axis=-1) % 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Theorem 2: k heads compute k-bit parity.")
    ap.add_argument("--bits", type=int, default=4, help="k, the number of input bits and heads")
    ap.add_argument("--mask", type=float, default=DEFAULT_MASK, help="finite M in equation (3)")
    args = ap.parse_args(argv)

    k = args.bits
    layer = parity_layer(k, mask=args.mask)
    xs = all_inputs(k)
    d_x, s_x = layer.stats(xs)
    delta = np.asarray(layer.delta(xs))
    F = np.asarray(layer.F(xs))
    out = np.asarray(layer.decide(xs))
    target = parity(xs)
    alpha, beta = partial_fractions(k)

    print(f"k = {k} heads, {k} bits, embedding dimension d = {k + 3}")
    print("Lemma 4 coefficients:")
    for j in range(k):
        print(f"  j={j + 1}  alpha={alpha[j]: .10f}  beta={beta[j]: .10f}")
    print()
    print(f"{'x':>{max(k, 5)}}  {'c':>2}  {'Delta(x)':>14}  {'P(c)/D(c)':>14}  {'F_x':>14}  out  par")
    for row in range(xs.shape[0]):
        bits = "".join(str(int(b)) for b in np.asarray(xs)[row])
        c = int(np.asarray(xs)[row].sum())
        print(
            f"{bits:>{max(k, 5)}}  {c:>2}  {delta[row]: .8e}  "
            f"{gap_at_weight(k, c): .8e}  {F[row]: .8e}   {out[row]}   {target[row]}"
        )
    print()
    print(f"denominators d_x^j on the all-zeros input: {np.asarray(d_x)[0]}")
    print(f"margin min_c |P(c)/D(c)| = {construction_margin(k):.6e}")
    print(f"parity solved on all {2 ** k} inputs: {bool((out == target).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
