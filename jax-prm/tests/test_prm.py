"""PRM module behaviour: memory init, grads, overfitting, baseline path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_prm.config import PRMConfig
from jax_prm.train import adamw_init, adamw_step, loss_fn
from jax_prm.transformer import count_params, forward, init


def _tiny(**kwargs) -> PRMConfig:
    base = dict(
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        vocab_size=32,
        context=16,
        n_slots=4,
        d_mem=16,
        n_mem_heads=4,
        dropout=0.0,
        aux_lambda=0.0,
        use_prm=True,
    )
    base.update(kwargs)
    return PRMConfig(**base)


def test_memory_init_zeros_then_updates():
    cfg = _tiny()
    params = init(jax.random.PRNGKey(0), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)
    _logits, info = forward(params, tokens, cfg)
    mem = info["memory"]
    assert mem is not None
    assert mem.shape == (2, cfg.n_slots, cfg.d_mem)
    # A fresh module starts from zeros; after a non-zero input it must move.
    assert float(jnp.max(jnp.abs(mem))) > 1e-6
    zeros_tokens = jnp.zeros((2, 8), dtype=jnp.int32)
    # Still: random embeddings of token 0 are not identically zero, so we only
    # assert the documented init is zeros by constructing memory ourselves.
    from jax_prm.prm import init_memory

    m0 = init_memory(3, cfg)
    np.testing.assert_array_equal(np.asarray(m0), np.zeros((3, cfg.n_slots, cfg.d_mem)))


def test_grads_flow_to_gru_and_gates():
    cfg = _tiny()
    params = init(jax.random.PRNGKey(7), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(8), (2, 8), 0, cfg.vocab_size)

    def _l(p):
        return loss_fn(p, tokens, cfg)

    grads = jax.grad(_l)(params)
    gru_g = grads["prm"]["gru"]
    for name in ("Wz", "Wr", "Wh", "bz", "br", "bh"):
        assert float(jnp.max(jnp.abs(gru_g[name]))) > 0.0, f"zero grad on gru.{name}"
    assert float(jnp.max(jnp.abs(grads["prm"]["Wg"]))) > 0.0
    assert float(jnp.max(jnp.abs(grads["prm"]["Wp"]))) > 0.0


def test_tiny_overfitting():
    cfg = _tiny()
    params = init(jax.random.PRNGKey(9), cfg)
    tokens = jax.random.randint(jax.random.PRNGKey(10), (2, 8), 0, cfg.vocab_size)
    opt = adamw_init(params)

    @jax.jit
    def step(params, opt, tokens):
        return adamw_step(params, opt, tokens, cfg, lr=1e-2, weight_decay=0.0)

    losses = []
    for _ in range(30):
        params, opt, loss = step(params, opt, tokens)
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_baseline_without_prm():
    cfg_off = _tiny(use_prm=False)
    cfg_slots = _tiny(n_slots=0)
    cfg_on = _tiny()
    key = jax.random.PRNGKey(11)
    p_off = init(key, cfg_off)
    p_slots = init(key, cfg_slots)
    p_on = init(key, cfg_on)
    tokens = jax.random.randint(jax.random.PRNGKey(12), (2, 8), 0, 32)
    logits_off, info_off = forward(p_off, tokens, cfg_off)
    logits_slots, info_slots = forward(p_slots, tokens, cfg_slots)
    assert logits_off.shape == (2, 8, 32)
    assert logits_slots.shape == (2, 8, 32)
    assert info_off["memory"] is None
    assert info_slots["memory"] is None
    assert "prm" not in p_off
    assert "prm" not in p_slots
    n_off = count_params(p_off)
    n_on = count_params(p_on)
    assert n_on > n_off
