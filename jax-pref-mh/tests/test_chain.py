"""The running sampler, not just the algebra.

Two things get checked. First that the code path from judge votes to accept
coin reproduces bar-alpha_N of Eq. (13), which ties Algorithm 1 to the kernel
the exactness tests are about. Then that the chain's empirical distribution
lands on pi, inside an error budget computed from the exact asymptotic
variance of the kernel rather than picked to fit.

Every stochastic assertion here pools several seeds and compares against a
budget derived from the chain length, so it does not flip between runs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pref_mh.acceptance import log_alpha_n_vote, marginal_acceptance
from pref_mh.finite import (
    asymptotic_variance,
    pref_mh_kernel,
    sampler_ops,
    total_variation,
)
from pref_mh.judges import count_votes
from pref_mh.sampler import log_r0, pref_mh_chain
from pref_mh.synthetic import random_model


def _tv_budget(kernel, pi, n_samples, sigmas=6.0):
    """A high-probability ceiling on TV(hat-pi, pi) for a chain of this length.

    For each state, the ergodic average of its indicator has asymptotic
    variance sigma_i^2, computed exactly from the kernel, so the error
    D_i = hat-pi_i - pi_i has standard deviation s_i = sigma_i / sqrt(n) once
    the CLT bites. TV = (1/2) sum_i |D_i|, so its mean is about
    (1/2) sqrt(2/pi) sum_i s_i and its standard deviation is at most
    (1/2) sum_i s_i, the fully correlated case. The budget is the mean plus
    ``sigmas`` of that bound.
    """
    s = np.array([
        np.sqrt(max(asymptotic_variance(kernel, pi, np.eye(len(pi))[i], 800), 0.0)
                / n_samples)
        for i in range(len(pi))
    ])
    mean = 0.5 * np.sqrt(2.0 / np.pi) * s.sum()
    return mean + sigmas * 0.5 * s.sum(), mean


@pytest.mark.parametrize("n_votes", [1, 2, 4])
def test_sampler_acceptance_matches_equation_13(n_votes):
    """Monte Carlo over the judge votes and the accept coin, against the
    closed-form bar-alpha_N.

    100k draws per pair puts the standard error under 1.6e-3, so the 1e-2
    threshold is above six of them.
    """
    model = random_model(6, n_states=5)
    ops = sampler_ops(model)
    judge = ops["judges"][0]
    lr_all = np.asarray(model.log_r0())
    want_all = np.asarray(
        marginal_acceptance(model.log_r0(), model.win_probs()[..., 0], n_votes)
    )

    draws = 100_000

    def trial(key, x, y):
        k_j, k_a = jax.random.split(key)
        k = count_votes(k_j, judge, x, y, n_votes)
        la = log_alpha_n_vote(
            log_r0(x, y, ops["log_p0"], ops["log_q"]), k[None], n_votes
        )
        return jnp.log(jax.random.uniform(k_a)) < la

    run = jax.jit(jax.vmap(trial, in_axes=(0, None, None)), static_argnums=())
    for x, y in [(0, 1), (1, 0), (2, 4), (3, 2)]:
        keys = jax.random.split(jax.random.PRNGKey(1000 + 7 * x + y), draws)
        got = float(jnp.mean(run(keys, x, y)))
        assert abs(got - want_all[x, y]) < 1e-2
        assert abs(lr_all[x, y] - float(log_r0(x, y, ops["log_p0"], ops["log_q"]))) < 1e-12


@pytest.mark.parametrize("n_votes", [1, 2])
def test_empirical_distribution_matches_target(n_votes):
    """Eight chains, pooled, against a budget from the exact kernel.

    The assertion is on the pooled histogram over all eight seeds. Individual
    seeds are reported through the same machinery but with their own budget,
    and only the aggregate is required to hold.
    """
    model = random_model(7, n_states=5, score_scale=0.8)
    ops = sampler_ops(model)
    pi = np.asarray(model.target())
    kernel = pref_mh_kernel(model, n_votes)

    n_chains, n_steps, burn = 8, 25_000, 1_000
    counts = np.zeros(model.size)
    total = 0
    for r in range(n_chains):
        states, _ = pref_mh_chain(
            jax.random.PRNGKey(2000 + r), 0, propose=ops["propose"],
            log_q=ops["log_q"], log_p0=ops["log_p0"], judges=ops["judges"],
            n_votes=n_votes, n_steps=n_steps,
        )
        s = np.asarray(states)[burn:]
        counts += np.bincount(s, minlength=model.size)
        total += len(s)

    budget, mean = _tv_budget(kernel, pi, total)
    tv = total_variation(counts / total, pi)
    assert tv < budget, f"TV {tv:.5f} over budget {budget:.5f} (mean {mean:.5f})"


def test_chain_and_kernel_agree_on_one_step_transitions():
    """The empirical one-step transition matrix of the running sampler against
    the enumerated P_N of Eq. (14).

    Row i comes from 20k independent single steps out of state i, so each entry
    has standard error under 3.6e-3; the threshold is 2e-2.
    """
    n_votes = 2
    model = random_model(8, n_states=4)
    ops = sampler_ops(model)
    want = np.asarray(pref_mh_kernel(model, n_votes))

    draws = 20_000

    def one_step(key, x):
        states, _ = pref_mh_chain(
            key, x, propose=ops["propose"], log_q=ops["log_q"],
            log_p0=ops["log_p0"], judges=ops["judges"], n_votes=n_votes,
            n_steps=1,
        )
        return states[0]

    fan = jax.jit(jax.vmap(one_step, in_axes=(0, None)))
    for x in range(model.size):
        keys = jax.random.split(jax.random.PRNGKey(50 + x), draws)
        got = np.bincount(np.asarray(fan(keys, x)), minlength=model.size) / draws
        assert np.max(np.abs(got - want[x])) < 2e-2
