"""SFT and weighted-SFT losses, plus g_0.

Symbols follow the paper: g_0(q) = grad_theta log pi_theta(o*|q), the SFT loss is
eq (1) with its uniform weight of 1, and the weighted loss is Algorithm 1
line 10.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .model import ModelConfig, log_probs

__all__ = [
    "expert_log_prob",
    "batch_expert_log_prob",
    "g0",
    "g0_sq_norms",
    "weighted_sft_loss",
    "sft_loss",
]


def expert_log_prob(params, config: ModelConfig, seq, prompt_len):
    """log pi_theta(o*|q) for one [q; o*] sequence, summed over the answer tokens."""
    return jnp.sum(log_probs(params, config, seq)[prompt_len - 1 :])


def batch_expert_log_prob(params, config: ModelConfig, batch, prompt_len):
    """log pi_theta(o*|q) for each row of a batch."""
    return jax.vmap(lambda s: expert_log_prob(params, config, s, prompt_len))(batch)


def g0(params, config: ModelConfig, seq, prompt_len):
    """g_0(q) = grad_theta log pi_theta(o*|q), the direction OSW-FT never touches."""
    return jax.grad(expert_log_prob)(params, config, seq, prompt_len)


def g0_sq_norms(params, config: ModelConfig, batch, prompt_len):
    """||g_0(q)||^2 per query, the factor that multiplies the weights in eq (11)."""
    def one(seq):
        grads = g0(params, config, seq, prompt_len)
        return sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))

    return jax.vmap(one)(batch)


@partial(jax.jit, static_argnames=("config", "prompt_len"))
def weighted_sft_loss(params, config: ModelConfig, batch, prompt_len, w):
    """Algorithm 1 line 10: L(theta) = -(1/|M|) sum w * log pi_theta(o*|q).

    w is treated as data. Remark 4 says why: differentiating through the weight
    would add a grad_theta p_theta(q) term that the estimator (1 - p_hat) g_0
    does not have, so the weights carry a stop-gradient and the loop optimizes
    the frozen surrogate J^(t) of Definition 1, eq (16).
    """
    w = jax.lax.stop_gradient(jnp.asarray(w))
    return -jnp.mean(w * batch_expert_log_prob(params, config, batch, prompt_len))


@partial(jax.jit, static_argnames=("config", "prompt_len"))
def sft_loss(params, config: ModelConfig, batch, prompt_len):
    """eq (1) and eq (8): the same loss with the uniform weight of 1."""
    return -jnp.mean(batch_expert_log_prob(params, config, batch, prompt_len))
