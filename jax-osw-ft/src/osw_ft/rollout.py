"""Inference-only rollouts and the success-rate estimate p_hat, Algorithm 1 lines 5-7.

Nothing in here is differentiated. The paper is explicit that rollouts feed a
scalar weight and never a gradient: the optimization direction stays anchored to
o*, only the magnitude moves.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr

from .model import ModelConfig, forward, log_probs
from .task import exact_match_reward
from .weighting import osw_weight, p_hat

__all__ = [
    "sample_answer",
    "sample_rollouts",
    "rollout_rewards",
    "estimate_p_hat",
    "estimate_weights",
    "success_probability",
]


@partial(jax.jit, static_argnames=("config", "answer_len"))
def sample_answer(params, config: ModelConfig, key, query, answer_len, temperature=1.0):
    """One rollout o_hat ~ pi_theta(.|q). Returns the (answer_len,) answer tokens."""
    prompt_len = query.shape[0]
    seq = jnp.concatenate([query, jnp.zeros((answer_len,), query.dtype)])
    for i in range(answer_len):
        logits = forward(params, config, seq)[prompt_len - 1 + i] / temperature
        key, subkey = jr.split(key)
        token = jr.categorical(subkey, logits).astype(seq.dtype)
        seq = seq.at[prompt_len + i].set(token)
    return seq[prompt_len:]


def sample_rollouts(params, config: ModelConfig, key, queries, answer_len, K, temperature=1.0):
    """K rollouts for each query in a batch. Returns (batch, K, answer_len)."""
    keys = jr.split(key, (queries.shape[0], K))
    per_query = jax.vmap(
        lambda q, ks: jax.vmap(
            lambda k: sample_answer(params, config, k, q, answer_len, temperature)
        )(ks)
    )
    return per_query(queries, keys)


def rollout_rewards(params, config: ModelConfig, key, batch, prompt_len, K, temperature=1.0):
    """Binary rewards R(o_hat_k) for K rollouts per query. Returns (batch, K)."""
    queries = batch[:, :prompt_len]
    expert = batch[:, prompt_len:]
    answers = sample_rollouts(
        params, config, key, queries, expert.shape[1], K, temperature
    )
    return exact_match_reward(answers, expert[:, None, :])


def estimate_p_hat(params, config: ModelConfig, key, batch, prompt_len, K, temperature=1.0):
    """p_hat(q) for each query, eq (9)."""
    return p_hat(rollout_rewards(params, config, key, batch, prompt_len, K, temperature))


def estimate_weights(params, config: ModelConfig, key, batch, prompt_len, K, temperature=1.0):
    """w = 1 - p_hat(q), Algorithm 1 line 7."""
    return osw_weight(rollout_rewards(params, config, key, batch, prompt_len, K, temperature))


@partial(jax.jit, static_argnames=("config", "prompt_len"))
def success_probability(params, config: ModelConfig, batch, prompt_len, temperature=1.0):
    """The exact p_s(q) = E_{o~pi}[R(o)|q] the rollouts are estimating.

    Only computable in closed form because the verifier accepts exactly one
    trajectory, so p_s(q) = pi_theta(o*|q) under the sampling temperature. Real
    tasks need the Monte Carlo estimate; tests use this to avoid asserting on
    sampling noise.
    """
    def one(seq):
        lp = log_probs(params, config, seq, temperature)
        return jnp.exp(jnp.sum(lp[prompt_len - 1 :]))

    return jax.vmap(one)(batch)
