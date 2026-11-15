"""The conservation law of Section 4.

If H_{k,h} really is the head's write into the residual stream, then the logit
difference has to be the sum of the embedding term, the MLP terms and every
Delta_{k,h}, with nothing left over. Anything adjacent to the real write, the
pre-projection value vector for instance, leaves a residue.
"""

import jax
import jax.numpy as jnp
import pytest

from head_influence.influence import deltas
from head_influence.model import (
    ModelConfig,
    forward,
    init_params,
    logit_direction,
    readout_bias_diff,
)
from head_influence.task import make_batch

from .conftest import DTYPE

RTOL = 1e-11


def reconstruct(cfg, params, tokens):
    trace = forward(cfg, params, tokens)
    w = logit_direction(params)
    pos = cfg.cls_index
    embed_term = w @ trace.embed[pos]
    mlp_term = (trace.mlp_out[:, pos, :] @ w).sum()
    head_term = deltas(trace.head_writes[:, :, pos, :], w).sum()
    total = embed_term + mlp_term + head_term + readout_bias_diff(params)
    return trace.logits[1] - trace.logits[0], total


@pytest.mark.parametrize("use_layernorm", [False, True])
def test_logit_difference_is_the_sum_of_its_parts(use_layernorm):
    cfg = ModelConfig(n_layers=3, n_heads=4, use_layernorm=use_layernorm, use_final_layernorm=False)
    params = init_params(jax.random.key(3), cfg, DTYPE)
    tokens, _ = make_batch(jax.random.key(4), 8, cfg.seq_len, cfg.vocab_size)
    for tok in tokens:
        got, rebuilt = reconstruct(cfg, params, tok)
        assert abs(got - rebuilt) <= RTOL * max(abs(got), 1.0)


def test_residual_stream_is_additive(random_params, cfg):
    """r^(L) = r^(0) + sum_k (A_k + MLP_k), position by position."""
    tokens, _ = make_batch(jax.random.key(5), 4, cfg.seq_len, cfg.vocab_size)
    for tok in tokens:
        trace = forward(cfg, random_params, tok)
        rebuilt = trace.embed + trace.attn_out.sum(0) + trace.mlp_out.sum(0)
        assert jnp.max(jnp.abs(rebuilt - trace.resid_final)) <= 1e-12 * jnp.max(
            jnp.abs(trace.resid_final))


def test_attention_output_is_the_sum_of_head_writes(random_params, cfg):
    """A_k = sum_h H_{k,h}, with no output bias hiding in the block."""
    tokens, _ = make_batch(jax.random.key(6), 4, cfg.seq_len, cfg.vocab_size)
    for tok in tokens:
        trace = forward(cfg, random_params, tok)
        assert jnp.max(jnp.abs(trace.head_writes.sum(1) - trace.attn_out)) <= 1e-13


def test_final_layernorm_breaks_the_decomposition():
    """The approximation the paper names when it abstracts normalization away.

    A LayerNorm between r^(L) and the readout makes the logit nonlinear in the
    residual state, so the decomposition stops being exact. The residual stream
    itself is still additive, so the error is entirely in the projection step.
    """
    cfg = ModelConfig(n_layers=3, n_heads=4, use_layernorm=True, use_final_layernorm=True)
    params = init_params(jax.random.key(7), cfg, DTYPE)
    tokens, _ = make_batch(jax.random.key(8), 4, cfg.seq_len, cfg.vocab_size)
    errors = []
    for tok in tokens:
        got, rebuilt = reconstruct(cfg, params, tok)
        errors.append(abs(float(got - rebuilt)))
        trace = forward(cfg, params, tok)
        rebuilt_resid = trace.embed + trace.attn_out.sum(0) + trace.mlp_out.sum(0)
        assert jnp.max(jnp.abs(rebuilt_resid - trace.resid_final)) <= 1e-12 * jnp.max(
            jnp.abs(trace.resid_final))
    assert min(errors) > 1e-3
