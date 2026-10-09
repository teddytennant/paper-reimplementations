"""Recursion, detach, shared block, decoder length."""

import jax
import jax.numpy as jnp
import path_thinker.thinker as thinker

from path_thinker.thinker import (
    answer_update,
    decode,
    init_thinker,
    latent_update,
    recurse,
    recurse_from,
    think_once,
)


def _net():
    params = init_thinker(jax.random.PRNGKey(0), d=8, y_len=5, z_len=3, c_skip=8, scale=0.2)
    c = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 8))
    return params, c


def test_block_call_count():
    params, c = _net()
    calls = {"n": 0}
    real = thinker.block

    def wrapped(params, tokens):
        calls["n"] += 1
        return real(params, tokens)

    thinker.block = wrapped
    try:
        recurse(params, c, n_sup=2, H=3, L=4)
    finally:
        thinker.block = real
    assert calls["n"] == 30


def test_incoming_y_is_detached():
    params, c = _net()
    y = jnp.broadcast_to(params["y0"], (c.shape[0],) + params["y0"].shape)
    z = jnp.broadcast_to(params["z0"], (c.shape[0],) + params["z0"].shape)

    def loss(marker):
        ys, _zs = recurse_from(params, c, y + marker, z, n_sup=1, H=1, L=1)
        return jnp.sum(ys[-1])

    g = jax.grad(loss)(jnp.zeros_like(y))
    assert jnp.allclose(g, 0.0)


def test_early_update_detached_last_update_trains():
    params, c = _net()
    y = jnp.broadcast_to(params["y0"], (c.shape[0],) + params["y0"].shape)
    z = jnp.broadcast_to(params["z0"], (c.shape[0],) + params["z0"].shape)
    marker0 = jnp.zeros_like(y)

    def loss_marker(marker):
        y_out, _z = think_once(
            params, c, y, z, H=2, L=1, inject_after_h={0: marker}
        )
        return jnp.sum(y_out)

    g = jax.grad(loss_marker)(marker0)
    assert jnp.allclose(g, 0.0)

    def loss_params(p):
        y_p = jnp.broadcast_to(p["y0"], y.shape)
        z_p = jnp.broadcast_to(p["z0"], z.shape)
        y_out, _z = think_once(p, c, y_p, z_p, H=2, L=1)
        return jnp.sum(y_out)

    gW = jax.grad(loss_params)(params)["Wq"]
    assert jnp.all(jnp.isfinite(gW))
    assert jnp.any(jnp.abs(gW) > 1e-8)


def test_shared_wq():
    params, c = _net()
    y = jnp.broadcast_to(params["y0"], (c.shape[0],) + params["y0"].shape)
    z = jnp.broadcast_to(params["z0"], (c.shape[0],) + params["z0"].shape)
    assert "Wq" in params
    assert not any(k.startswith("Wq_") for k in params)

    def lat(W):
        p = dict(params)
        p["Wq"] = W
        return jnp.sum(latent_update(p, c, y, z))

    def ans(W):
        p = dict(params)
        p["Wq"] = W
        return jnp.sum(answer_update(p, y, z))

    gl = jax.grad(lat)(params["Wq"])
    ga = jax.grad(ans)(params["Wq"])
    assert jnp.any(gl != 0)
    assert jnp.any(ga != 0)


def test_decoder_uses_spatial_length_only():
    params = init_thinker(jax.random.PRNGKey(2), d=8, y_len=7, z_len=2, c_skip=4, scale=0.2)
    y = jax.random.normal(jax.random.PRNGKey(3), (1, 7, 8))
    seen = {}
    real = thinker.project_tokens

    def wrapped(params, tokens):
        seen["len"] = tokens.shape[1]
        return real(params, tokens)

    thinker.project_tokens = wrapped
    try:
        out = decode(params, y, n_spatial=4, height=2, width=2)
    finally:
        thinker.project_tokens = real
    assert seen["len"] == 4
    assert y.shape[1] == 7
    assert out.shape == (1, 2, 2, 4)
