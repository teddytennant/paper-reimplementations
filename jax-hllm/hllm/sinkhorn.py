"""Sinkhorn relaxation of the assignment, and the distillation loss.

Section 3.3. The Hungarian decoder returns a discrete permutation and has no
gradient, so training replaces it with the Sinkhorn operator (Cuturi, 2013;
Mena et al., 2018), which projects a non-negative matrix onto the Birkhoff
polytope by alternating row and column normalization. Equation (2):

    S^(0)   = exp(M / tau),
    S^(l+1) = D_r^-1 D_c^-1 S^(l),

with D_r and D_c the diagonal matrices that force unit row and column sums and
tau > 0 the temperature. As tau -> 0 the plan approaches a permutation matrix.

Equation (3) is the training objective, a cross entropy against the teacher's
permutation matrix P_teacher in {0, 1}^{N x K}:

    L = - sum_{i,j} P_teacher_{ij} log S^(L)_{ij}.

Two notes on what the paper leaves open.

Equation (2) writes both normalizations as left multiplications, which cannot
be literally right for the column one; the usual reading, and the one here, is
one row pass then one column pass per iteration. Which pass comes last matters
for a finite l: after a column pass the columns sum to exactly 1 and the rows
only approximately. :func:`log_sinkhorn` ends on the row pass, so its rows are
exact and its columns converge; the difference washes out well before the
iteration counts used in training.

Equation (3) does not say whether S is normalized as a joint plan or per row.
Doubly stochastic S has row sums of 1, so each row is already a distribution
over positions and the objective is the sum of N per-item cross entropies. That
is how it is implemented, and it is why the loss floor is 0.

Everything here runs in log space. exp(M / tau) at the small tau the paper
wants underflows in float32 almost immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["log_sinkhorn", "sinkhorn", "permutation_cross_entropy", "sinkhorn_loss"]


def log_sinkhorn(scores, tau: float = 1.0, n_iters: int = 20):
    """Equation (2) in log space. Returns ``log S^(L)``, shape ``(N, N)``.

    ``scores`` is the score matrix M. The initialization ``S^(0) = exp(M/tau)``
    becomes ``log S^(0) = M / tau``, and each normalization becomes a
    ``logsumexp`` subtraction, which is the same fixed point without the
    overflow.
    """
    log_s = jnp.asarray(scores) / tau
    if log_s.ndim != 2 or log_s.shape[0] != log_s.shape[1]:
        raise ValueError(
            f"Sinkhorn needs a square score matrix, got shape {log_s.shape}. "
            "Only the Hungarian decoder handles N != K."
        )

    def step(log_s, _):
        log_s = log_s - jax.scipy.special.logsumexp(log_s, axis=0, keepdims=True)
        log_s = log_s - jax.scipy.special.logsumexp(log_s, axis=1, keepdims=True)
        return log_s, None

    log_s, _ = jax.lax.scan(step, log_s, None, length=n_iters)
    return log_s


def sinkhorn(scores, tau: float = 1.0, n_iters: int = 20):
    """Equation (2) as a plan in probability space, ``S^(L)``."""
    return jnp.exp(log_sinkhorn(scores, tau=tau, n_iters=n_iters))


def permutation_cross_entropy(log_plan, teacher):
    """Equation (3), given ``log S^(L)`` and a teacher permutation matrix.

    ``teacher`` is P_teacher, one-hot over positions in each row. The sum is
    divided by N so the number reads as a mean per item and does not drift with
    slate size; the gradient direction is unchanged.
    """
    log_plan = jnp.asarray(log_plan)
    teacher = jnp.asarray(teacher, dtype=log_plan.dtype)
    n = log_plan.shape[0]
    return -jnp.sum(teacher * log_plan) / n


def sinkhorn_loss(scores, teacher, tau: float = 1.0, n_iters: int = 20):
    """Equations (2) and (3) end to end, from M and P_teacher to a scalar."""
    return permutation_cross_entropy(
        log_sinkhorn(scores, tau=tau, n_iters=n_iters), teacher
    )
