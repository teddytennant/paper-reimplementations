"""Tiny discrete flow-matching CE trainer on synthetic tokens."""

import jax

from flow_duality.dfm import train_tiny


def test_tiny_dfm_loss_drops():
    loss0, loss1, _ = train_tiny(
        jax.random.PRNGKey(7),
        n_steps=200,
        batch_size=256,
        seq_len=4,
        vocab_size=2,
        hidden=32,
        lr=0.03,
        eval_size=1024,
    )
    loss0 = float(loss0)
    loss1 = float(loss1)
    assert loss1 < loss0, (loss0, loss1)
