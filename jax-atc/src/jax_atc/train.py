"""AdamW and the ATC / data-curriculum training loops."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_atc.data import sample_parity
from jax_atc.model import (
    ATCConfig,
    build_cots,
    empty_cots,
    forward,
    mse_loss,
    sign_accuracy,
)


def adamw_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adamw_update(
    params,
    grads,
    state,
    lr: float = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd: float = 0.0,
):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1.0 - b2) * jnp.square(g), state["v"], grads
    )
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps) - lr * wd * p,
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


def make_step(
    config: ATCConfig,
    phase: int,
    lr: float = 1e-2,
    wd: float = 0.0,
    data_curriculum_only: bool = False,
):
    """JIT-friendly train step. ATC uses ``phase-1`` CoT tokens; the baseline uses none."""
    n_cots = 0 if data_curriculum_only else (phase - 1)
    head_index = 0 if data_curriculum_only else None
    loss_phase = 1 if data_curriculum_only else phase

    def step(params, opt_state, bits, y):
        def loss_fn(p):
            if n_cots == 0:
                cots = empty_cots(bits.shape[0], config)
            else:
                cots = build_cots(p, bits, n_cots, config)
            return mse_loss(
                p, bits, cots, y, loss_phase, config, head_index=head_index
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt_state = adamw_update(params, grads, opt_state, lr=lr, wd=wd)
        return params, opt_state, loss

    return jax.jit(step)


def evaluate(
    params,
    bits,
    y,
    phase: int,
    config: ATCConfig,
    data_curriculum_only: bool = False,
):
    n_cots = 0 if data_curriculum_only else (phase - 1)
    head_index = 0 if data_curriculum_only else None
    loss_phase = 1 if data_curriculum_only else phase
    if n_cots == 0:
        cots = empty_cots(bits.shape[0], config)
    else:
        cots = build_cots(params, bits, n_cots, config)
    res = forward(
        params, bits, cots, loss_phase, config, head_index=head_index
    )
    loss = jnp.mean((res.yhat - y) ** 2)
    acc = sign_accuracy(res.yhat, y)
    return loss, acc


def train_phase(
    params,
    key,
    phase: int,
    config: ATCConfig,
    *,
    lr: float = 1e-2,
    wd: float = 0.0,
    batch_size: int = 64,
    max_steps: int = 400,
    loss_threshold: float = 0.02,
    subset=None,
    data_curriculum_only: bool = False,
    eval_every: int = 20,
    opt_state=None,
):
    """Train on D_phase until eval MSE < ``loss_threshold`` or ``max_steps``."""
    if opt_state is None:
        opt_state = adamw_init(params)
    step = make_step(
        config,
        phase,
        lr=lr,
        wd=wd,
        data_curriculum_only=data_curriculum_only,
    )
    key, eval_key = jax.random.split(key)
    eval_bits, eval_y = sample_parity(
        eval_key, config.n, phase, batch_size, subset=subset
    )
    last_loss = jnp.inf
    last_acc = 0.0
    steps_done = 0
    for t in range(max_steps):
        key, dkey = jax.random.split(key)
        bits, y = sample_parity(dkey, config.n, phase, batch_size, subset=subset)
        params, opt_state, _ = step(params, opt_state, bits, y)
        steps_done = t + 1
        if (t + 1) % eval_every == 0 or t == max_steps - 1:
            last_loss, last_acc = evaluate(
                params,
                eval_bits,
                eval_y,
                phase,
                config,
                data_curriculum_only=data_curriculum_only,
            )
            last_loss = float(last_loss)
            last_acc = float(last_acc)
            if last_loss < loss_threshold:
                break
    return params, opt_state, {
        "loss": last_loss,
        "accuracy": last_acc,
        "steps": steps_done,
        "phase": phase,
    }


def run_curriculum(
    params,
    key,
    config: ATCConfig,
    n_phases: int,
    **train_kwargs,
):
    """Train phases 1..n_phases in order, appending a CoT token after each phase."""
    opt_state = adamw_init(params)
    history = []
    for r in range(1, n_phases + 1):
        key, pkey = jax.random.split(key)
        params, opt_state, info = train_phase(
            params,
            pkey,
            r,
            config,
            opt_state=opt_state,
            **train_kwargs,
        )
        history.append(info)
    return params, history
