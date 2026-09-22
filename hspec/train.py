"""Train a tiny target on a repeating motif, then a frozen-target drafter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax.numpy as jnp

from hspec.draft import (
    D,
    DEFAULT_GAMMA,
    N_HEADS,
    N_LAYERS,
    VOCAB,
    D_HEAD,
    acceptance_length,
    drafter_forward_batch,
    init_drafter,
    init_target,
    last_hiddens,
    sample_motif_batch,
    target_forward,
    target_greedy_tokens,
)


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _clip(grads, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def _adam_update(params, grads, state, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    grads = _clip(grads)
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1**t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2**t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat
    )
    return new_params, {"m": m, "v": v, "t": t}


def _target_loss(params, tokens):
    logits, _, _ = target_forward(params, tokens)
    logp = jax.nn.log_softmax(logits[:, :-1, :])
    gathered = jnp.take_along_axis(logp, tokens[:, 1:, None], axis=-1)
    return -jnp.mean(gathered)


def _prefix_features(target_params, tokens, prefix_len):
    logits, hiddens, kvs = target_forward(target_params, tokens)
    h_last = last_hiddens(hiddens, prefix_len - 1)
    target_k = jnp.concatenate([k[:, :prefix_len] for k, _ in kvs], axis=1)
    target_v = jnp.concatenate([v[:, :prefix_len] for _, v in kvs], axis=1)
    teacher = jax.lax.stop_gradient(logits[:, prefix_len - 1 : prefix_len - 1 + DEFAULT_GAMMA, :])
    return h_last, target_k, target_v, teacher


def _drafter_loss(drafter_params, target_params, tokens, prefix_len):
    h_last, target_k, target_v, teacher = _prefix_features(target_params, tokens, prefix_len)
    h_last = jax.lax.stop_gradient(h_last)
    target_k = jax.lax.stop_gradient(target_k)
    target_v = jax.lax.stop_gradient(target_v)
    draft_logits = drafter_forward_batch(
        drafter_params, h_last, target_k, target_v, prefix_len, DEFAULT_GAMMA
    )
    logp = jax.nn.log_softmax(draft_logits)
    prob = jax.nn.softmax(teacher)
    return -jnp.mean(jnp.sum(prob * logp, axis=-1))


def _eval_acceptance(target_params, drafter_params, key, prefix_len, n=64):
    tokens = sample_motif_batch(key, n, prefix_len + DEFAULT_GAMMA)
    prefix = tokens[:, :prefix_len]
    h_last, target_k, target_v, _ = _prefix_features(target_params, tokens, prefix_len)
    hybrid = drafter_forward_batch(
        drafter_params, h_last, target_k, target_v, prefix_len, DEFAULT_GAMMA, True, True
    )
    ablate = drafter_forward_batch(
        drafter_params,
        h_last,
        target_k,
        target_v,
        prefix_len,
        DEFAULT_GAMMA,
        inject_state=False,
        inject_kv=False,
    )
    target_tokens = target_greedy_tokens(target_params, prefix, DEFAULT_GAMMA)
    hybrid_len = acceptance_length(jnp.argmax(hybrid, axis=-1), target_tokens)
    ablate_len = acceptance_length(jnp.argmax(ablate, axis=-1), target_tokens)
    return float(jnp.mean(hybrid_len)), float(jnp.mean(ablate_len))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    key, kt, kd = jax.random.split(key, 3)
    target_params = init_target(kt)
    drafter_params = init_drafter(kd)
    target_opt = _adam_init(target_params)
    drafter_opt = _adam_init(drafter_params)

    batch = 64
    seq_len = 24
    prefix_len = 8
    target_lr = 2e-2
    drafter_lr = 2e-2

    @jax.jit
    def target_step(params, opt, key):
        tokens = sample_motif_batch(key, batch, seq_len)
        loss, grads = jax.value_and_grad(_target_loss)(params, tokens)
        params, opt = _adam_update(params, grads, opt, target_lr)
        return params, opt, loss

    @jax.jit
    def drafter_step(dparams, dopt, tparams, key):
        tokens = sample_motif_batch(key, batch, prefix_len + DEFAULT_GAMMA)
        loss, grads = jax.value_and_grad(_drafter_loss)(dparams, tparams, tokens, prefix_len)
        dparams, dopt = _adam_update(dparams, grads, dopt, drafter_lr)
        return dparams, dopt, loss

    key, k0 = jax.random.split(key)
    target_loss0 = float(_target_loss(target_params, sample_motif_batch(k0, batch, seq_len)))
    print(f"target_loss_step0 {target_loss0}")
    target_loss_last = target_loss0
    for step in range(args.steps):
        key, ks = jax.random.split(key)
        target_params, target_opt, loss = target_step(target_params, target_opt, ks)
        target_loss_last = float(loss)
        if step == 0 or step + 1 == args.steps:
            print(f"target_loss_step {step} {target_loss_last}")
    print(f"target_loss_last {target_loss_last}")

    key, k0 = jax.random.split(key)
    tokens0 = sample_motif_batch(k0, batch, prefix_len + DEFAULT_GAMMA)
    drafter_loss0 = float(_drafter_loss(drafter_params, target_params, tokens0, prefix_len))
    print(f"loss_step0 {drafter_loss0}")
    drafter_loss_last = drafter_loss0
    for step in range(args.steps):
        key, ks = jax.random.split(key)
        drafter_params, drafter_opt, loss = drafter_step(drafter_params, drafter_opt, target_params, ks)
        drafter_loss_last = float(loss)
        if step == 0 or step + 1 == args.steps:
            print(f"drafter_loss_step {step} {drafter_loss_last}")
    print(f"loss_last {drafter_loss_last}")

    key, ke = jax.random.split(key)
    mean_h, mean_a = _eval_acceptance(target_params, drafter_params, ke, prefix_len, n=128)
    print(f"mean_accepted_hybrid {mean_h}")
    print(f"mean_accepted_no_injection {mean_a}")

    results = {
        "steps": args.steps,
        "seed": args.seed,
        "loss_step0": drafter_loss0,
        "loss_last": drafter_loss_last,
        "target_loss_step0": target_loss0,
        "target_loss_last": target_loss_last,
        "mean_accepted_hybrid": mean_h,
        "mean_accepted_no_injection": mean_a,
        "d": D,
        "vocab": VOCAB,
        "n_layers": N_LAYERS,
        "gamma": DEFAULT_GAMMA,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
