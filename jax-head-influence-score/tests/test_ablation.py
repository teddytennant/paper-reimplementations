"""Ablation, the causal machinery behind Section 6.4.

Two things are being checked. First, that Delta_{k,h} is exactly the head's
direct contribution to the logit difference: remove the write from the final
residual state and the logit difference drops by exactly Delta. Second, that
deleting the write and rerunning the model perturbs nothing it should not,
that is, no term of the decomposition upstream of the ablated head moves.
"""

import jax
import jax.numpy as jnp

from head_influence.ablation import direct_path_logit_diff, head_mask, logit_diff
from head_influence.influence import deltas
from head_influence.model import ModelConfig, forward, init_params, logit_direction
from head_influence.task import make_batch

from .conftest import DTYPE


def _setup(use_layernorm=False, seed=11):
    cfg = ModelConfig(n_layers=3, n_heads=4, use_layernorm=use_layernorm)
    params = init_params(jax.random.key(seed), cfg, DTYPE)
    tokens, _ = make_batch(jax.random.key(seed + 1), 4, cfg.seq_len, cfg.vocab_size)
    return cfg, params, tokens


def test_direct_path_ablation_removes_exactly_delta():
    """For every head, not one of them."""
    cfg, params, tokens = _setup()
    w = logit_direction(params)
    for tok in tokens:
        trace = forward(cfg, params, tok)
        base = trace.logits[1] - trace.logits[0]
        delta = deltas(trace.head_writes[:, :, cfg.cls_index, :], w)
        for k in range(cfg.n_layers):
            for h in range(cfg.n_heads):
                got = direct_path_logit_diff(cfg, params, tok, [(k, h)])
                assert abs(float(got - (base - delta[k, h]))) <= 1e-12 * max(
                    abs(float(base)), 1.0)


def test_full_ablation_zeroes_the_head_and_leaves_upstream_alone():
    """Rerunning with the write deleted moves only what sits downstream of it."""
    cfg, params, tokens = _setup()
    w = logit_direction(params)
    pos = cfg.cls_index
    for tok in tokens:
        clean = forward(cfg, params, tok)
        clean_delta = deltas(clean.head_writes[:, :, pos, :], w)
        for k in range(cfg.n_layers):
            for h in range(cfg.n_heads):
                abl = forward(cfg, params, tok, head_mask(cfg, [(k, h)], DTYPE))
                abl_delta = deltas(abl.head_writes[:, :, pos, :], w)
                assert abs(float(abl_delta[k, h])) == 0.0
                # Everything the ablated head cannot reach is untouched.
                assert jnp.allclose(abl_delta[:k], clean_delta[:k], rtol=0, atol=1e-13)
                same_layer = [i for i in range(cfg.n_heads) if i != h]
                assert jnp.allclose(abl_delta[k, same_layer], clean_delta[k, same_layer],
                                    rtol=0, atol=1e-13)
                assert jnp.allclose(abl.mlp_out[:k, pos], clean.mlp_out[:k, pos],
                                    rtol=0, atol=1e-13)
                assert jnp.allclose(abl.embed, clean.embed, rtol=0, atol=1e-13)


def test_full_ablation_of_a_last_layer_head_matches_delta_when_mlps_are_off():
    """With nothing downstream to react, the rerun agrees with the direct path.

    Zeroing the MLP writes leaves a last-layer head with only the readout below
    it, so deleting it and rerunning has to cost exactly Delta_{k,h}. Earlier
    layers still differ, because they feed the attention patterns above them.
    """
    cfg, params, tokens = _setup()
    params = dict(params)
    params["mlp"] = dict(params["mlp"])
    params["mlp"]["w2"] = jnp.zeros_like(params["mlp"]["w2"])
    params["mlp"]["b2"] = jnp.zeros_like(params["mlp"]["b2"])
    w = logit_direction(params)
    last = cfg.n_layers - 1
    reacted = []
    for tok in tokens:
        trace = forward(cfg, params, tok)
        base = float(trace.logits[1] - trace.logits[0])
        delta = deltas(trace.head_writes[:, :, cfg.cls_index, :], w)
        for h in range(cfg.n_heads):
            abl = forward(cfg, params, tok, head_mask(cfg, [(last, h)], DTYPE))
            got = float(abl.logits[1] - abl.logits[0])
            assert abs(got - (base - float(delta[last, h]))) <= 1e-12 * max(abs(base), 1.0)
        for h in range(cfg.n_heads):
            abl = forward(cfg, params, tok, head_mask(cfg, [(0, h)], DTYPE))
            got = float(abl.logits[1] - abl.logits[0])
            reacted.append(abs(got - (base - float(delta[0, h]))))
    assert max(reacted) > 1e-6


def test_ablation_is_consistent_on_the_trained_model(trained):
    """Same identity on weights that were actually fit, not just sampled."""
    from .conftest import to_float64

    cfg, params, acc = trained
    params = to_float64(params)
    tokens, _ = make_batch(jax.random.key(21), 8, cfg.seq_len, cfg.vocab_size)
    w = logit_direction(params)
    for tok in tokens:
        trace = forward(cfg, params, tok)
        base = trace.logits[1] - trace.logits[0]
        delta = deltas(trace.head_writes[:, :, cfg.cls_index, :], w)
        for k in range(cfg.n_layers):
            for h in range(cfg.n_heads):
                got = direct_path_logit_diff(cfg, params, tok, [(k, h)])
                assert abs(float(got - (base - delta[k, h]))) <= 1e-11 * max(
                    abs(float(base)), 1.0)


def test_masking_no_heads_is_a_no_op(random_params, cfg):
    tokens, _ = make_batch(jax.random.key(31), 4, cfg.seq_len, cfg.vocab_size)
    a = logit_diff(cfg, random_params, tokens)
    b = logit_diff(cfg, random_params, tokens, head_mask(cfg, [], DTYPE))
    assert jnp.allclose(a, b, rtol=0, atol=1e-13)
