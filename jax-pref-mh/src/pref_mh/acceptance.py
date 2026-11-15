"""Acceptance rules from "When Metropolis and Hastings Meet Bradley and Terry:
Exact MCMC From Preference Voting" (arXiv:2609.00905).

Map from the paper:

* Eq. (5)  ``bt_win_prob``            p_J(x < x') = sigma(s(x') - s(x))
* Eq. (6)  ``bt_log_odds``            the win-rate ratio, equal to e^{s(x')-s(x)}
* Eq. (7)  ``log_alpha_bt_mh``        the oracle BT-MH rule, not implementable
* Eq. (8)  ``log_alpha_n_vote``       the Pref-MH N-vote rule (single judge)
* Eq. (12) ``log_alpha_n_vote``       the multi-judge rule, m > 1
* Eq. (13) ``marginal_acceptance``    bar-alpha_N, averaged over the judge votes
* App. C.1 ``log_alpha_plugin``       the invalid plug-in baseline

Everything here works in log space. ``r_0(x,x') = p_0(x') q(x|x') /
(p_0(x) q(x'|x))`` is passed as ``log_r0`` because it is an unnormalized
density ratio and can be large; ``min{1, ...}`` becomes ``min{0, ...}`` before
the exponential, so nothing overflows. K = 0 gives ``log(0) = -inf``, which
flows through to an acceptance probability of exactly zero, matching the
paper's remark that a zero vote count always rejects.
"""

import math

import jax.numpy as jnp
from jax.scipy.special import expit


def bt_win_prob(score_x, score_y):
    """p_J(x < y) under the Bradley-Terry model, Eq. (5).

    The probability that the judge prefers y to x, sigma(s(y) - s(x)).
    """
    return expit(score_y - score_x)


def bt_log_odds(win_prob):
    """log of the win-rate ratio p/(1-p), Eq. (6).

    The paper's key observation: this equals s(y) - s(x), so the BT odds are
    the log of the judge's likelihood ratio p_J(M|y)/p_J(M|x), which is the
    judge-dependent half of the MH ratio in Eq. (3).
    """
    return jnp.log(win_prob) - jnp.log1p(-win_prob)


def log_alpha_bt_mh(log_r0, win_prob):
    """Oracle BT-MH acceptance rule, Eq. (7): min{1, r_0 * p/(1-p)}.

    Theorem 1 says no procedure that sees only N binary votes can reproduce
    this for all p in (0,1). It is here as the thing Pref-MH does not do, and
    for the oracle baseline that gets the latent score directly.
    """
    return jnp.minimum(0.0, log_r0 + bt_log_odds(win_prob))


def log_vote_factor(k, n_votes):
    """log of the N-vote factor K/(N-K+1) from Eq. (8).

    K = N gives factor N, the largest upweighting; K = 0 gives 0, a certain
    rejection; a split vote gives roughly 1, so r_0 decides.
    """
    k = jnp.asarray(k)
    return jnp.log(k) - jnp.log(n_votes - k + 1)


def log_alpha_n_vote(log_r0, counts, n_votes):
    """Pref-MH acceptance rule, Eq. (8) for one judge and Eq. (12) for m.

    ``counts`` holds K_i, the number of the N votes from judge i that favour
    the proposal. Pass a scalar for the single-judge rule or a length-m array
    for the multi-judge rule; the m factors multiply, which is what makes
    adding a condition mean adding a judge.
    """
    counts = jnp.asarray(counts)
    log_factor = jnp.sum(log_vote_factor(counts, n_votes), axis=-1)
    return jnp.minimum(0.0, log_r0 + log_factor)


def log_alpha_plugin(log_r0, k, n_votes):
    """The invalid plug-in baseline of Appx. C.1.

    Estimates the odds as K/(N-K) and drops that into Eq. (7). K = 0 rejects
    and K = N accepts outright, as the paper specifies. Theorem 1 rules this
    out as an implementation of the oracle rule, and it does not satisfy
    detailed balance either, so its chain has the wrong stationary
    distribution.
    """
    k = jnp.asarray(k)
    log_odds_hat = jnp.log(k) - jnp.log(n_votes - k)
    log_alpha = jnp.minimum(0.0, log_r0 + log_odds_hat)
    return jnp.where(k == n_votes, 0.0, jnp.where(k == 0, -jnp.inf, log_alpha))


def _log_binom_pmf(n_votes, p):
    """log Binomial(N, p) pmf over k = 0..N, last axis.

    The binomial coefficients come from ``math.comb`` rather than a lgamma
    call, so they are exact for the budgets used here (C(N,k) < 2^53).
    """
    p = jnp.asarray(p)[..., None]
    k = jnp.arange(n_votes + 1, dtype=p.dtype)
    log_coef = jnp.asarray(
        [math.log(math.comb(n_votes, int(i))) for i in range(n_votes + 1)],
        dtype=p.dtype,
    )
    return log_coef + k * jnp.log(p) + (n_votes - k) * jnp.log1p(-p)


def marginal_acceptance(log_r0, win_prob, n_votes):
    """bar-alpha_N(x,y) of Eq. (13), the acceptance probability of Eq. (8)
    averaged over K ~ Binomial(N, p_J(x < y)).

    This is the quantity detailed balance is stated on (Eq. 9), and the
    off-diagonal entry of the marginal kernel P_N of Eq. (14). Computed by
    exhaustive summation over k, so it carries no Monte Carlo error.
    """
    k = jnp.arange(n_votes + 1)
    log_alpha = jnp.minimum(
        0.0, jnp.asarray(log_r0)[..., None] + log_vote_factor(k, n_votes)
    )
    return jnp.sum(jnp.exp(_log_binom_pmf(n_votes, win_prob) + log_alpha), axis=-1)


def marginal_acceptance_multi(log_r0, win_probs, n_votes):
    """bar-alpha_N^{(m)}(x,y): Eq. (13) for the m-judge rule of Eq. (12).

    ``win_probs`` is shape (..., m). The vote counts are independent across
    judges given x and y, so the average is a sum over the full (N+1)^m grid
    of count vectors. Only used for exact tests with small m and N.
    """
    win_probs = jnp.asarray(win_probs)
    m = win_probs.shape[-1]
    k = jnp.arange(n_votes + 1)
    grids = jnp.stack(jnp.meshgrid(*([k] * m), indexing="ij"), axis=-1)
    grids = grids.reshape(-1, m)  # (G, m)

    log_factor = jnp.sum(log_vote_factor(grids, n_votes), axis=-1)  # (G,)
    log_alpha = jnp.minimum(0.0, jnp.asarray(log_r0)[..., None] + log_factor)

    log_pmf = _log_binom_pmf(n_votes, win_probs)  # (..., m, N+1)
    log_w = jnp.zeros_like(log_alpha)
    for i in range(m):
        log_w = log_w + log_pmf[..., i, :][..., grids[:, i]]
    return jnp.sum(jnp.exp(log_w + log_alpha), axis=-1)


def marginal_acceptance_plugin(log_r0, win_prob, n_votes):
    """The plug-in rule of Appx. C.1 averaged over K ~ Binomial(N, p).

    Same shape as Eq. (13) but with the plug-in realized rule. Used to show
    the plug-in chain misses detailed balance and lands on a different
    stationary distribution.
    """
    k = jnp.arange(n_votes + 1)
    log_alpha = log_alpha_plugin(jnp.asarray(log_r0)[..., None], k, n_votes)
    return jnp.sum(jnp.exp(_log_binom_pmf(n_votes, win_prob) + log_alpha), axis=-1)


def marginal_acceptance_restricted(log_r0, win_prob, n_votes, allowed):
    """An exact competitor to Eq. (8) that throws away some vote counts.

    Uses the N-vote acceptance probability on counts in ``allowed`` and zero
    on the rest. The proof of Theorem 3 shows an exact rule's per-count
    acceptance b_k must satisfy b_k C(N,k) = c b_{N-k+1} C(N,N-k+1), so
    zeroing a count k forces zeroing N-k+1 as well. Whenever ``allowed`` is
    closed under k -> N-k+1 the identity still holds termwise and the
    competitor is exact, just uniformly less willing to move. That makes it a
    real, non-trivial rule to test Peskun-Tierney domination against.
    """
    k = jnp.arange(n_votes + 1)
    keep = jnp.isin(k, jnp.asarray(allowed))
    log_alpha = jnp.minimum(
        0.0, jnp.asarray(log_r0)[..., None] + log_vote_factor(k, n_votes)
    )
    log_alpha = jnp.where(keep, log_alpha, -jnp.inf)
    return jnp.sum(jnp.exp(_log_binom_pmf(n_votes, win_prob) + log_alpha), axis=-1)
