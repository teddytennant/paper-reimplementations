"""Theorem 1: the alternating-sum obstruction, made numerical.

Clearing the k strictly positive softmax denominators turns the decision of a
k-head layer into the sign of

    F_x = sum_j s_x^j prod_{j' != j} d_x^{j'}.

Every monomial of F_x is a product of exactly k factors, each of which reads at
most one input position, so on n > k bits some coordinate x_{i*} is always
missing and the inner sum over x_{i*} in {0,1} of (-1)^{x_{i*}} kills it. Hence

    sum_{x in {0,1}^n} sigma_x F_x = 0,     sigma_x = (-1)^{x_1 + ... + x_n},

for every choice of parameters whenever n > k. Solving n-bit parity would force
every sigma_x F_x >= 0 with at least one strictly positive, so k heads cannot do
it. At n = k the identity is not available, and the Theorem 2 construction makes
the sum strictly positive; `parity_alternating_sum` shows both sides of that.
"""

from __future__ import annotations

import numpy as np

from head_complexity.layer import (
    AttentionCore,
    ArgmaxLayer,
    JointEmbedding,
    all_inputs,
)
from head_complexity.parity import parity_layer

import jax.numpy as jnp


def sigma(xs) -> np.ndarray:
    """sigma_x = (-1)^{sum_i x_i}, the parity sign of Section 3.1."""
    return np.where(np.asarray(xs).sum(axis=-1) % 2 == 0, 1.0, -1.0)


def alternating_sum(layer: ArgmaxLayer, n: int | None = None) -> tuple[float, float]:
    """(sum_x sigma_x F_x, sum_x |F_x|) over the whole cube {0,1}^n.

    The second number is the natural scale for the first: F_x carries the
    product of k denominators, so an absolute tolerance on the sum alone would
    mean nothing. Claim 1 of Theorem 1 says the ratio is zero for n > k.
    """
    if n is None:
        n = layer.core.n
    xs = all_inputs(n)
    F = np.asarray(layer.F(xs), dtype=np.float64)
    return float((sigma(xs) * F).sum()), float(np.abs(F).sum())


def relative_alternating_sum(layer: ArgmaxLayer, n: int | None = None) -> float:
    """|sum_x sigma_x F_x| / sum_x |F_x|, the quantity Claim 1 sends to zero."""
    total, scale = alternating_sum(layer, n)
    return abs(total) / scale


def parity_alternating_sum(k: int, n: int) -> float:
    """The obstruction ratio for the Theorem 2 k-head layer run on n bits.

    n = k is the construction itself, where every sigma_x F_x is strictly
    positive and the sum has to be too. n = k + 1 is the case Theorem 1 rules
    out: the same k heads on one more bit, where the sum collapses to zero no
    matter how the parameters were chosen.
    """
    return relative_alternating_sum(parity_layer(k, n=n), n=n)


def random_argmax_layer(
    seed: int,
    k: int,
    n: int,
    d: int = 5,
    d_out: int = 3,
    score_scale: float = 0.35,
) -> ArgmaxLayer:
    """A k-head joint-embedding layer with random parameters.

    Theorem 1 constrains nothing about Phi, A_j, V_j or W_O, so a random draw is
    a fair witness for Claim 1. `score_scale` keeps the entries of A_j small
    enough that the raw scores xi_j . v_t, and so the r's that `head_stats`
    exponentiates, stay well inside float64 range; the identity itself is scale
    free, since rescaling head j's r's by a positive constant only multiplies
    F_x by a constant that does not depend on x.
    """
    rng = np.random.default_rng(seed)
    emb = JointEmbedding(
        jnp.asarray(rng.normal(size=(2, n, d)), dtype=jnp.float64),
        jnp.asarray(rng.normal(size=(d,)), dtype=jnp.float64),
    )
    core = AttentionCore(
        emb,
        jnp.asarray(score_scale * rng.normal(size=(k, d, d)), dtype=jnp.float64),
        jnp.asarray(rng.normal(size=(k, d, d_out)), dtype=jnp.float64),
    )
    return ArgmaxLayer(core, jnp.asarray(rng.normal(size=(2, d_out)), dtype=jnp.float64))


__all__ = [
    "alternating_sum",
    "parity_alternating_sum",
    "random_argmax_layer",
    "relative_alternating_sum",
    "sigma",
]
