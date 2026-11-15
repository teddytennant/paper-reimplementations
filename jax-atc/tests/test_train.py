"""Tiny ATC loop (existence, not Fig. 2) and data-curriculum-only smoke."""

import jax
import jax.numpy as jnp

from jax_atc.model import ATCConfig, init_params
from jax_atc.train import evaluate, train_phase


def test_tiny_train_two_phases():
    n, d = 4, 32
    config = ATCConfig.tiny(n=n, d=d)
    key = jax.random.PRNGKey(0)
    key, pkey = jax.random.split(key)
    params = init_params(pkey, config)

    key, t1 = jax.random.split(key)
    params, _, info1 = train_phase(
        params,
        t1,
        phase=1,
        config=config,
        lr=1e-2,
        batch_size=64,
        max_steps=300,
        loss_threshold=0.02,
        eval_every=20,
    )
    assert info1["accuracy"] > 0.90, info1

    key, t2 = jax.random.split(key)
    params, _, info2 = train_phase(
        params,
        t2,
        phase=2,
        config=config,
        lr=1e-2,
        batch_size=64,
        max_steps=400,
        loss_threshold=0.02,
        eval_every=20,
    )
    assert info2["accuracy"] > 0.70, info2


def test_data_curriculum_only_smoke():
    n, d = 4, 32
    config = ATCConfig.tiny(n=n, d=d)
    key = jax.random.PRNGKey(1)
    key, pkey = jax.random.split(key)
    params = init_params(pkey, config)
    key, tkey = jax.random.split(key)
    params, _, info = train_phase(
        params,
        tkey,
        phase=2,
        config=config,
        lr=1e-2,
        batch_size=32,
        max_steps=8,
        loss_threshold=0.0,
        eval_every=4,
        data_curriculum_only=True,
    )
    assert info["loss"] == info["loss"]  # not NaN
    assert jnp.isfinite(jnp.asarray(info["loss"]))
    # one more eval through the public API
    bits = jnp.ones((8, n)).at[:, :2].set(
        jnp.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
    )
    y = bits[:, 0] * bits[:, 1]
    loss, acc = evaluate(
        params, bits, y, phase=2, config=config, data_curriculum_only=True
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(acc)
