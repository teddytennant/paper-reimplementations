"""Tiny AXON encoder shapes and classification head."""

import jax
import jax.numpy as jnp

from aaa.model import axon_classify, axon_encode, init_axon_params


def test_encoder_output_shape():
    key = jax.random.PRNGKey(0)
    b, c, t, d = 2, 4, 8, 32
    x = jax.random.normal(key, (b, c, t, d))
    params = init_axon_params(
        jax.random.PRNGKey(1), c=c, t=t, d=d, num_blocks=2, gate="gated"
    )
    out = axon_encode(params, x, num_heads=4)
    assert out.shape == (b, c, t, d)


def test_classify_logits_shape():
    key = jax.random.PRNGKey(2)
    b, c, t, d, k = 3, 4, 5, 16, 2
    x = jax.random.normal(key, (b, c, t, d))
    params = init_axon_params(
        jax.random.PRNGKey(3),
        c=c,
        t=t,
        d=d,
        num_blocks=2,
        num_classes=k,
        gate="fixed",
    )
    logits = axon_classify(params, x, num_heads=4)
    assert logits.shape == (b, k)


def test_fixed_lambda_shared_inside_layer():
    from aaa.gate import fixed_lambda

    params = init_axon_params(
        jax.random.PRNGKey(4), c=3, t=4, d=8, num_blocks=2, gate="fixed"
    )
    for block in params["blocks"]:
        lam = fixed_lambda(block["gate"])
        assert lam.shape == ()
        assert 0.0 < float(lam) < 1.0
