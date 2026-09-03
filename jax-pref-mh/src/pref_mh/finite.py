"""Exhaustive machinery for finite state spaces.

On a finite X we can write down the marginal kernel P_N of Eq. (14) entry by
entry, because bar-alpha_N of Eq. (13) is a finite sum over vote counts. That
turns the paper's claims into linear algebra: detailed balance is Eq. (9) held
against a matrix, and stationarity is pi @ P = pi. Nothing here samples, so
none of it carries Monte Carlo error.

Conventions: ``log_p0`` is shape (S,) up to an additive constant, ``log_q`` is
shape (S, S) with ``log_q[j, i] = log q(j | i)`` (column i is the proposal
distribution from state i), and ``scores`` is shape (m, S), one latent score
per judge.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .acceptance import (
    bt_win_prob,
    marginal_acceptance,
    marginal_acceptance_multi,
    marginal_acceptance_plugin,
    marginal_acceptance_restricted,
)


@dataclass(frozen=True)
class DiscreteModel:
    """A finite-space instance: base density, proposal, and judge scores."""

    log_p0: jnp.ndarray  # (S,)
    log_q: jnp.ndarray  # (S, S), log_q[j, i] = log q(j | i)
    scores: jnp.ndarray  # (m, S)

    @property
    def size(self):
        return self.log_p0.shape[0]

    @property
    def n_judges(self):
        return self.scores.shape[0]

    def log_target(self):
        """log pi up to a constant: the Gibbs form of Eq. (4), extended to m
        judges by Eq. (10), log p_0 + sum_i s_i."""
        return self.log_p0 + jnp.sum(self.scores, axis=0)

    def target(self):
        """pi, normalized."""
        lt = self.log_target()
        w = jnp.exp(lt - jnp.max(lt))
        return w / jnp.sum(w)

    def log_r0(self):
        """(S, S) matrix of log r_0(i, j), the tractable factor of Eq. (3)."""
        lp = self.log_p0
        lq = self.log_q
        return (lp[None, :] + lq) - (lp[:, None] + lq.T)

    def win_probs(self):
        """(S, S, m) with entry [i, j, c] = p_{J_c}(i < j), Eq. (5)."""
        s = self.scores  # (m, S)
        return jnp.moveaxis(bt_win_prob(s[:, :, None], s[:, None, :]), 0, -1)


def kernel_from_acceptance(model, abar):
    """Assemble P from the marginal acceptance matrix, Eq. (14).

    Off-diagonal entries are q(j|i) bar-alpha(i, j); the diagonal absorbs
    everything else, both self-proposals and rejections.
    """
    q = jnp.exp(model.log_q).T  # q[i, j] = q(j | i)
    off = q * abar
    off = off.at[jnp.diag_indices(model.size)].set(0.0)
    return off + jnp.diag(1.0 - jnp.sum(off, axis=1))


def pref_mh_kernel(model, n_votes):
    """P_N for Pref-MH, Eq. (8) / Eq. (12) averaged as in Eq. (13)."""
    lr = model.log_r0()
    w = model.win_probs()
    if model.n_judges == 1:
        abar = marginal_acceptance(lr, w[..., 0], n_votes)
    else:
        abar = marginal_acceptance_multi(lr, w, n_votes)
    return kernel_from_acceptance(model, abar)


def oracle_mh_kernel(model):
    """P for standard MH on the true density, Eq. (2)."""
    lt = model.log_target()
    lq = model.log_q
    ratio = (lt[None, :] + lq) - (lt[:, None] + lq.T)
    return kernel_from_acceptance(model, jnp.exp(jnp.minimum(0.0, ratio)))


def plugin_kernel(model, n_votes):
    """P for the invalid plug-in sampler of Appx. C.1. Single judge only."""
    assert model.n_judges == 1, "the plug-in baseline is defined for one judge"
    abar = marginal_acceptance_plugin(
        model.log_r0(), model.win_probs()[..., 0], n_votes
    )
    return kernel_from_acceptance(model, abar)


def restricted_kernel(model, n_votes, allowed):
    """P for the exact competitor that ignores vote counts outside ``allowed``.

    See ``acceptance.marginal_acceptance_restricted``. Used to test the
    Peskun-Tierney claim of Theorem 3.
    """
    assert model.n_judges == 1
    abar = marginal_acceptance_restricted(
        model.log_r0(), model.win_probs()[..., 0], n_votes, allowed
    )
    return kernel_from_acceptance(model, abar)


def stationary(kernel):
    """The stationary vector of a finite kernel, by a left null-space solve.

    Solves pi (P - I) = 0 with the normalization sum pi = 1 in place of one
    redundant equation, rather than iterating powers, so the answer does not
    depend on a mixing rate.
    """
    p = np.asarray(kernel, dtype=np.float64)
    n = p.shape[0]
    a = np.vstack([(p - np.eye(n)).T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return sol


def total_variation(p, q):
    """TV distance between two probability vectors, as in Appx. C.1."""
    return 0.5 * float(np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def asymptotic_variance(kernel, pi, f, n_terms=2000):
    """Asymptotic variance of the ergodic average of f, by summing
    autocovariances: sigma^2 = gamma_0 + 2 sum_{t>=1} gamma_t.

    Peskun-Tierney domination implies this is no larger for the dominating
    kernel, for every f. Summed directly rather than through a fundamental
    matrix so there is no formula to get wrong; the small chains used here
    decay geometrically well inside ``n_terms``.
    """
    p = np.asarray(kernel, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64) - float(np.dot(pi, f))
    d = pi * f
    total = float(np.dot(d, f))
    v = f.copy()
    for _ in range(n_terms):
        v = p @ v
        total += 2.0 * float(np.dot(d, v))
    return total


def sampler_ops(model):
    """Wrap a ``DiscreteModel`` as the callables ``sampler.py`` expects.

    States are integer indices. The judges built here are Bradley-Terry judges
    over the model's latent scores, so the sampler sees only votes while the
    tests can still compare against the exact pi.
    """
    from .judges import bradley_terry_judge

    log_p0 = model.log_p0
    log_q = model.log_q
    scores = model.scores
    log_target = model.log_target()

    def propose(key, x):
        import jax

        return jax.random.categorical(key, log_q[:, x])

    def log_q_fn(x_to, x_from):
        return log_q[x_to, x_from]

    def log_p0_fn(x):
        return log_p0[x]

    def log_target_fn(x):
        return log_target[x]

    judges = tuple(
        bradley_terry_judge(lambda x, row=scores[i]: row[x])
        for i in range(model.n_judges)
    )
    return {
        "propose": propose,
        "log_q": log_q_fn,
        "log_p0": log_p0_fn,
        "log_target": log_target_fn,
        "judges": judges,
    }
