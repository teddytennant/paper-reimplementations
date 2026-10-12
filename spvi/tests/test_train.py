"""Smoke training step and no-event forecast on a tiny cyclic batch."""

import jax
import jax.numpy as jnp

from spvi.flow import flow
from spvi.model import Config, init_params, jump_coefficients
from spvi.train import forecast, make_optimizer, synthetic_cyclic_batch, train_step


def _any_nonzero(tree):
    leaves = jax.tree.leaves(tree)
    return any(bool(jnp.any(jnp.abs(leaf) > 0)) for leaf in leaves)


def test_jump_net_reads_only_marks_and_lower_blocks():
    config = Config()
    params = init_params(jax.random.PRNGKey(0), config)
    marks = jnp.asarray([[[0.2, -0.4], [0.1, 0.3]]])
    empty = jnp.zeros((1, 2, 0))
    r0, b0 = jump_coefficients(params, 0, marks, empty, empty)
    r0b, _ = jump_coefficients(params, 0, marks + 0.5, empty, empty)
    assert r0.shape == (1, 2, 1)
    assert jnp.all(r0 > 1e-2) and jnp.all(r0 < 1.0)
    assert jnp.any(jnp.abs(r0 - r0b) > 1e-6)

    lower_pre = jnp.asarray([[[-0.2], [0.4]]])
    lower_post = jnp.asarray([[[0.1], [-0.3]]])
    r1, b1 = jump_coefficients(params, 1, marks, lower_pre, lower_post)
    r1b, b1b = jump_coefficients(params, 1, marks, lower_pre, lower_post + 0.7)
    assert jnp.all(r1 > 1e-2) and jnp.all(r1 < 1.0)
    assert jnp.any(jnp.abs(r1 - r1b) + jnp.abs(b1 - b1b) > 1e-6)


def test_train_step_and_forecast():
    config = Config(l_max=4, n_quad=1, hidden=4, embed_dim=4)
    key = jax.random.PRNGKey(0)
    k_init, k_data, k_step, k_fore = jax.random.split(key, 4)
    params = init_params(k_init, config)
    batch = synthetic_cyclic_batch(k_data, batch_size=2, n_times=12, noise=0.1, config=config)
    assert batch["times"].shape == (2, 12)
    assert batch["y"].shape == (2, 12, 2)
    context = {"times": batch["times"][:, :8], "y": batch["y"][:, :8]}
    future_times = batch["times"][:, 8:]
    optimizer = make_optimizer(1e-3)
    opt_state = optimizer.init(params)

    losses = []
    grads = None
    for i in range(3):
        k_step, k_i = jax.random.split(k_step)
        params, opt_state, loss, grads = train_step(
            params, opt_state, optimizer, context, k_i, config
        )
        losses.append(float(loss))
        assert jnp.isfinite(loss)

    assert _any_nonzero(grads["jump"])
    assert _any_nonzero(grads["rate_head"])

    out = forecast(params, context["times"], context["y"], future_times, k_fore, config)
    assert out["future_states"].shape == (2, 4, 2)
    assert jnp.all(jnp.isfinite(out["future_states"]))

    # No context events and no future candidates: forecast is the pure flow.
    empty = jnp.zeros((2, 0))
    k_fore, k_none = jax.random.split(k_fore)
    none = forecast(
        params,
        context["times"],
        context["y"],
        future_times,
        k_none,
        config,
        include_context_events=False,
        future_offsets=empty,
    )
    horizon = context["times"][:, -1]
    assert jnp.allclose(
        none["z_T"],
        flow(none["z0"], horizon, none["gamma"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert jnp.allclose(
        none["future_states"],
        flow(none["z0"][:, None, :], future_times, none["gamma"]),
        atol=1e-5,
        rtol=1e-5,
    )
    # From the stored terminal state the same no-event query is the flow over the gap.
    assert jnp.allclose(
        none["future_states"],
        flow(none["z_T"][:, None, :], future_times - horizon[:, None], none["gamma"]),
        atol=1e-5,
        rtol=1e-5,
    )
