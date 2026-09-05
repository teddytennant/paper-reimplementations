"""Differentiable sum-product belief propagation for the outer code.

Equations (3), (4), (5) and (6) of the paper, plus the arctanh stabilization of
Remark 1. Everything is written in dense (n_checks x n) message matrices masked
by H, which is wasteful for a real LDPC code and perfectly fine at the block
lengths used here, and which keeps the whole thing a single jittable function
with no scatter/gather.

Sign convention. The paper defines its LLR in (2) as

    L_i = log( Pr(c_i = 1 | y) / Pr(c_i = 0 | y) )

and the BCE loss in (36) reads sig(c_hat_BP,i) as Pr(c_i = 1), so the convention
is fixed: positive LLR means bit 1. The message updates (3) and (4) as printed
are the textbook sum-product rules, which are self-consistent only in the
opposite convention, log(p0/p1); applying (4) literally to log(p1/p0) messages
multiplies every check-to-variable message by (-1)^(d-1). The two conventions
agree on even-degree checks and disagree on odd-degree ones. We take (2) and
(36) as authoritative, so `decode` accepts and returns log(p1/p0) LLRs and runs
the message passing in the log(p0/p1) form internally. `check_to_variable`
below is exported separately for anyone who wants (4) applied verbatim.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .ops import DEFAULT_EPS, masked_prod_excluding, stable_arctanh


def check_to_variable(
    mu_vc: jax.Array, mask: jax.Array, eps: float = DEFAULT_EPS
) -> jax.Array:
    """Equation (4), verbatim, on a dense (n_checks, n) message matrix.

        mu_cv[j, i] = 2 arctanh( prod_{b in N_v(j) \\ i} tanh( mu_vc[j, b] / 2 ) )

    Entries outside the support of H are returned as 0.
    """
    t = jnp.tanh(mu_vc / 2.0)
    prod = masked_prod_excluding(t, mask, axis=-1)
    return jnp.where(mask, 2.0 * stable_arctanh(prod, eps), 0.0)


def variable_to_check(
    llr: jax.Array, mu_cv: jax.Array, mask: jax.Array
) -> jax.Array:
    """Equation (3) on a dense (n_checks, n) message matrix.

        mu_vc[j, i] = L_i + sum_{k in N_c(i) \\ j} mu_cv[k, i]

    The leave-one-out sum is the column sum minus the term itself, which is
    exact here because addition, unlike the product in (4), always inverts.
    """
    col = jnp.sum(jnp.where(mask, mu_cv, 0.0), axis=-2, keepdims=True)
    return jnp.where(mask, llr[..., None, :] + col - mu_cv, 0.0)


def decode(
    llr: jax.Array,
    h: np.ndarray | jax.Array,
    n_iter: int,
    eps: float = DEFAULT_EPS,
) -> jax.Array:
    """Equation (6): c_hat_out = BP(c_hat_in; N_BP).

    `llr` is (..., n) in the paper's log(p1/p0) convention, `n_iter` is N_BP.
    n_iter == 0 returns the input unchanged, which is the "inner-code-only
    training" case the paper calls N_BP = 0.

    The loop is a Python for, not lax.scan, on purpose: N_BP is a static
    hyperparameter here and unrolling is what makes the growth of the product of
    Jacobians in (37) directly visible to reverse-mode autodiff, which is the
    thing the gradient-explosion test measures.
    """
    mask = jnp.asarray(np.asarray(h) == 1)
    if n_iter == 0:
        return llr
    internal = -llr  # log(p1/p0) -> log(p0/p1)
    mu_cv = jnp.zeros(jnp.broadcast_shapes(internal.shape[:-1] + (1, 1), mask.shape))
    for _ in range(n_iter):
        mu_vc = variable_to_check(internal, mu_cv, mask)
        mu_cv = check_to_variable(mu_vc, mask, eps)
    posterior = internal + jnp.sum(jnp.where(mask, mu_cv, 0.0), axis=-2)  # eq (5)
    return -posterior


def hard_decision(llr: jax.Array) -> jax.Array:
    """Bit estimates from log(p1/p0) LLRs: 1 where the LLR is positive."""
    return (llr > 0).astype(jnp.uint8)


def ml_decode(llr: jax.Array, codewords: np.ndarray) -> jax.Array:
    """Exhaustive maximum-likelihood decoding over an enumerated codebook.

    Given per-bit channel LLRs L_i = log(p1/p0) from a memoryless channel, the
    log-likelihood of a codeword c is sum_i c_i L_i up to a constant that does
    not depend on c, so the ML codeword is the argmax of that inner product.
    Ties are broken by taking the lowest index, which matters only on a measure
    zero set of inputs.

    `codewords` is (n_cw, n); returns the winning codeword per batch element.
    """
    cw = jnp.asarray(codewords, dtype=llr.dtype)
    scores = llr @ cw.T
    return jnp.asarray(codewords)[jnp.argmax(scores, axis=-1)]


def app_decode_llr(llr: jax.Array, codewords: np.ndarray) -> jax.Array:
    """Exhaustive bitwise-MAP (a posteriori probability) LLRs over a codebook.

    This, not blockwise ML, is what sum-product BP is trying to compute. For a
    memoryless channel with per-bit LLRs L (log p1/p0) and a uniform prior over
    the codebook,

        L'_i = log( sum_{c in C, c_i = 1} exp(<c, L>) )
             - log( sum_{c in C, c_i = 0} exp(<c, L>) ).

    On a cycle-free Tanner graph BP reproduces these numbers exactly once it has
    run for at least the graph diameter; on a graph with cycles it does not, and
    the gap is the whole reason BP is called an approximation.
    """
    cw = jnp.asarray(codewords, dtype=llr.dtype)
    scores = llr @ cw.T  # (..., n_cw)
    is_one = cw > 0.5  # (n_cw, n)
    big = jnp.finfo(llr.dtype).min / 4
    s1 = jax.scipy.special.logsumexp(
        jnp.where(is_one.T[None, ...], scores[..., None, :], big), axis=-1
    )
    s0 = jax.scipy.special.logsumexp(
        jnp.where(~is_one.T[None, ...], scores[..., None, :], big), axis=-1
    )
    return jnp.reshape(s1 - s0, llr.shape)
