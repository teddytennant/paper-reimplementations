"""The OSW-FT weight and the exact identities the paper proves about it.

Everything here is closed form. For a query q with true success probability
p_s(q), the method draws K inference-only rollouts, sets

    p_hat(q) = (1/K) sum_k R(o_k),        w(q) = 1 - p_hat(q),

and scales that query's SFT loss by w. See eq (9) and Algorithm 1.

The functions below state the paper's results as formulas; the tests prove them
by enumerating the K+1 binomial outcomes with exact rational arithmetic instead
of sampling, since every claim in Theorem 1, Remark 1, Remark 2 and eq (11) is
an exact statement about a finite sum.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jr

__all__ = [
    "p_hat",
    "osw_weight",
    "hard_threshold_weight",
    "random_weight",
    "weight_pmf",
    "expected_weight",
    "expected_squared_weight",
    "signal_loss_probability",
    "estimation_mse",
    "sigma_squared",
    "sigma_squared_sft",
    "optimal_control_variate_mse",
]


def p_hat(rewards):
    """Monte Carlo success rate from K binary rollout rewards, eq (9).

    `rewards` is shaped (..., K) with entries in {0, 1}. Averaging is over the
    last axis, so a batch of queries goes through in one call.
    """
    return jnp.mean(jnp.asarray(rewards), axis=-1)


def osw_weight(rewards):
    """w(q) = 1 - p_hat(q), the trajectory-level weight of eq (9)."""
    return 1.0 - p_hat(rewards)


def hard_threshold_weight(rewards):
    """Ablation control from section 4.4: w = 0 iff all K rollouts succeed.

    This is the discrete version of the weight, equivalent to hard rejection
    sampling. It keeps the signal-loss behaviour of eq (13) but throws away the
    intermediate values that Remark 1 says are the point of K >= 2.
    """
    return (p_hat(rewards) < 1.0).astype(jnp.result_type(float))


def random_weight(key, mean_weight, shape):
    """Ablation control from section 4.4: weights decoupled from the model.

    Uniform draws calibrated to have exactly `mean_weight`, so the comparison
    against OSW-FT isolates content-aware weighting from a plain learning rate
    decay. The support is [max(0, 2m-1), min(1, 2m)], which is centred on m and
    stays inside [0, 1] for any m in [0, 1].
    """
    m = jnp.clip(jnp.asarray(mean_weight), 0.0, 1.0)
    lo = jnp.maximum(0.0, 2.0 * m - 1.0)
    hi = jnp.minimum(1.0, 2.0 * m)
    return jr.uniform(key, shape, minval=lo, maxval=hi)


def weight_pmf(p, K):
    """Exact distribution of w = 1 - p_hat over the K+1 binomial outcomes.

    Returns (values, probs) where values[m] = 1 - m/K is the weight when m of
    the K rollouts succeed and probs[m] = C(K, m) p^m (1-p)^(K-m). Useful when
    you want an expectation over the estimator without sampling it.
    """
    m = jnp.arange(K + 1)
    coef = jnp.asarray([math.comb(K, i) for i in range(K + 1)], dtype=float)
    p = jnp.asarray(p, dtype=float)
    probs = coef * p**m * (1.0 - p) ** (K - m)
    return 1.0 - m / K, probs


def expected_weight(p, K=None):
    """Theorem 1: E[w] = 1 - p for any finite K, so E[w * g_0] = (1-p) g_0.

    K is accepted and ignored, which is the content of the theorem: the OSW-FT
    estimator is unbiased for the exact surrogate update at every rollout count,
    not just in the limit.
    """
    del K
    return 1.0 - jnp.asarray(p)


def expected_squared_weight(p, K):
    """Remark 2, eq (14): E[w^2] = (1-p)^2 + p(1-p)/K.

    Signal term plus an estimation-noise term that dies off as 1/K.
    """
    p = jnp.asarray(p)
    return (1.0 - p) ** 2 + p * (1.0 - p) / K


def signal_loss_probability(p, K):
    """Remark 1, eq (13): P(w = 0) = p^K, since w = 0 iff every rollout is correct.

    This is what makes K = 2 the interesting point. At K = 1 a query with
    p = 0.5 drops out half the time and the weight is binary; at K = 2 the loss
    probability is 0.25 and w = 0.5 becomes reachable.
    """
    return jnp.asarray(p) ** K


def estimation_mse(p, g0_sq_norm, K):
    """Remark 3, eq (15): E||g_hat_K - (1-p) g_0||^2 = p(1-p)/K * ||g_0||^2."""
    p = jnp.asarray(p)
    return p * (1.0 - p) / K * jnp.asarray(g0_sq_norm)


def sigma_squared(p, g0_sq_norm, K):
    """eq (11): Sigma^2 = E_q[((1-p)^2 + p(1-p)/K) ||g_0||^2].

    The stochastic-gradient second moment that sets the L*Sigma^2 term in the
    Theorem 2 rate. `p` and `g0_sq_norm` are per-query arrays; the expectation
    over q is the sample mean.
    """
    return jnp.mean(expected_squared_weight(p, K) * jnp.asarray(g0_sq_norm))


def sigma_squared_sft(g0_sq_norm):
    """eq (11): Sigma^2_SFT = E_q[||g_0||^2], the uniform-weight baseline.

    Corollary 1 gives Sigma^2_SFT - Sigma^2 >= E_q[p(q) ||g_0(q)||^2] for every
    K >= 1, so the gap is strict as soon as p > 0 on a positive-measure set.
    """
    return jnp.mean(jnp.asarray(g0_sq_norm))


def optimal_control_variate_mse(p, c):
    """Section 2.1, eq (4): E_o[(R - c)^2] for binary R with P(R=1) = p.

    Equals p(1-c)^2 + (1-p)c^2, minimized at c* = p = p_s(q). That minimizer is
    also the advantage of the oracle trajectory, A(o*, q) = 1 - p_s(q) in
    eq (7), which is where the weight comes from.
    """
    p = jnp.asarray(p)
    c = jnp.asarray(c)
    return p * (1.0 - c) ** 2 + (1.0 - p) * c**2
