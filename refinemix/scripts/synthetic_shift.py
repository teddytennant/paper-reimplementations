#!/usr/bin/env python
"""CPU synthetic domain shift for RefineMix.

Target and source sequences share a local arithmetic pattern and differ by a
marker token. Trains target-only, RefineMix, and data mixing for a small step
count and reports the held-out target NELBO (equation 13) and the fraction of
samples whose marker is the source marker.

This does not reproduce the paper's tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _device_from_argv(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cpu")
    known, _ = parser.parse_known_args(argv)
    return known.device


def _build_parser():
    parser = argparse.ArgumentParser(description="Synthetic RefineMix domain shift")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=16)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    device = _device_from_argv(argv)
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        os.environ["JAX_PLATFORMS"] = device

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import jax
    import jax.numpy as jnp
    import optax

    from refinemix.algorithm import make_train_step
    from refinemix.denoiser import init_denoiser, make_apply_fn
    from refinemix.kernels import noise_tokens
    from refinemix.loss import masked_nelbo
    from refinemix.sampler import ancestral_sample
    from refinemix.schedule import EPS, sample_time

    args = _build_parser().parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    length = 8
    pattern = 8
    target_marker = pattern
    source_marker = pattern + 1
    # Tokens 0..7 are the shared alphabet, 8 and 9 are markers, 10 and 11 are
    # unused, 12 is the mask. Twelve content ids plus a mask.
    vocab_size = 13
    mask_id = 12

    def make_sequence(marker, start):
        seq = [marker]
        token = start % pattern
        for _ in range(length - 1):
            seq.append(token)
            token = (token + 1) % pattern
        return seq

    train_starts = [0, 1, 2, 3, 4]
    holdout_starts = [5, 6, 7]
    target_train = jnp.asarray(
        [make_sequence(target_marker, s) for s in train_starts], dtype=jnp.int32
    )
    source_train = jnp.asarray(
        [make_sequence(source_marker, s) for s in range(pattern)], dtype=jnp.int32
    )
    holdout = jnp.asarray(
        [make_sequence(target_marker, s) for s in holdout_starts], dtype=jnp.int32
    )

    modes = {
        "target_only": {"tmax": 0.0, "lam": 0.0},
        "refinemix": {"tmax": 0.5, "lam": 0.5},
        "data_mixing": {"tmax": 1.0, "lam": 0.5},
    }

    apply_fn = make_apply_fn(n_heads=2, mask_id=mask_id)
    optimizer = optax.adamw(1e-3, weight_decay=0.01)
    train_step = make_train_step(apply_fn, optimizer, mask_id)

    def take_batch(pool, key):
        idx = jax.random.randint(key, (args.batch_size,), 0, pool.shape[0])
        return pool[idx]

    def eval_nelbo(params, key):
        losses = []
        for i in range(4):
            k_t, k_n = jax.random.split(jax.random.fold_in(key, i))
            t = sample_time(k_t, (holdout.shape[0],), mode="experimental")
            zt = noise_tokens(holdout, t, k_n, kernel="masked", mask_id=mask_id)
            logits = apply_fn(params, zt, t)
            losses.append(masked_nelbo(logits, holdout, zt, t, mask_id))
        return float(jnp.mean(jnp.stack(losses)))

    report_modes = {}
    for index, (name, cfg) in enumerate(modes.items()):
        key = jax.random.PRNGKey(args.seed + 17 * index)
        k_init, key = jax.random.split(key)
        params, _ = init_denoiser(
            k_init,
            vocab_size=vocab_size,
            seq_len=length,
            d_model=32,
            n_heads=2,
            mask_id=mask_id,
        )
        opt_state = optimizer.init(params)
        final_loss = None
        for step_i in range(args.steps):
            key, k_step, k_tgt, k_src = jax.random.split(key, 4)
            target_batch = take_batch(target_train, k_tgt)
            source_batch = take_batch(source_train, k_src)
            params, opt_state, loss, _, _ = train_step(
                params,
                opt_state,
                target_batch,
                source_batch,
                k_step,
                jnp.asarray(cfg["tmax"]),
                jnp.asarray(cfg["lam"]),
            )
            final_loss = loss
        key, k_eval, k_sample = jax.random.split(key, 3)
        nelbo = eval_nelbo(params, k_eval)
        samples = ancestral_sample(
            apply_fn,
            params,
            k_sample,
            batch_size=args.n_samples,
            seq_len=length,
            n_steps=args.sample_steps,
            mask_id=mask_id,
        )
        fraction = float(jnp.mean(samples[:, 0] == source_marker))
        report_modes[name] = {
            "tmax": cfg["tmax"],
            "lam": cfg["lam"],
            "target_nelbo": nelbo,
            "source_marker_fraction": fraction,
            "final_train_loss": float(final_loss),
        }

    report = {
        "steps": args.steps,
        "device": device,
        "seed": args.seed,
        "seq_len": length,
        "vocab_size": vocab_size,
        "mask_id": mask_id,
        "target_marker": target_marker,
        "source_marker": source_marker,
        "time_sampler": "uniform(eps, 1)",
        "eps": EPS,
        "modes": report_modes,
        "note": (
            "Small model and step count. Reports equation 13 on held-out "
            "target sequences with t in (eps, 1). Does not reproduce the paper tables."
        ),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
