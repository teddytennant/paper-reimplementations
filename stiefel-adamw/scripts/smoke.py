#!/usr/bin/env python3
"""Tiny factorization smoke run for Stiefel-AdamW (no pretrained models)."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from stiefel_adamw.geometry import random_stiefel, row_orth_error
from stiefel_adamw.optimizer import factorization_loss, init, step_from_dW


def main(steps: int = 300, m: int = 64, n: int = 64, r: int = 8) -> None:
    key = jax.random.key(0)
    k_b, k_a, k_w = jax.random.split(key, 3)
    B = jax.random.normal(k_b, (m, r))
    A = random_stiefel(k_a, n, r).T
    W = jax.random.normal(k_w, (m, r)) @ random_stiefel(jax.random.key(1), n, r).T
    state = init(B, A, project_a=False)

    def body(state, _):
        dW = state.B @ state.A - W
        state = step_from_dW(
            state,
            dW,
            lr=1e-2,
            weight_decay=0.0,
            bias_correction=True,
            retraction="cayley",
            n_cayley_iters=6,
        )
        return state, factorization_loss(state.B, state.A, W)

    state, losses = jax.lax.scan(body, state, None, length=steps)
    payload = {
        "loss": float(losses[-1]),
        "loss0": float(factorization_loss(B, A, W)),
        "max_orth_err": float(row_orth_error(state.A)),
        "steps": steps,
        "m": m,
        "n": n,
        "r": r,
        "device": str(jax.default_backend()),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
