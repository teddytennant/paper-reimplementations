"""Tiny CPU train on the paper's 4-token deterministic grammar task."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from jax_vqt.data import decode_tokens, grammar_score, make_dataset, token_accuracy, windows
from jax_vqt.model import VQTConfig, count_params, init_params, parameter_count
from jax_vqt.train import adam_init, greedy_generate, make_step


def main():
    cfg = VQTConfig(n_encoder=2, n_decoder=2, n_var_layers=1)
    train, test = make_dataset("deterministic")
    x, y = windows(train)
    params = init_params(jax.random.PRNGKey(0), cfg)
    n = count_params(params)
    print(f"model {cfg.name} params {n} (formula {parameter_count(2, 2, 1)})")
    print(f"train tokens {len(train)} windows {len(x)}")

    step = make_step(cfg, lr=1e-2)
    opt = adam_init(params)
    x_j, y_j = jnp.asarray(x), jnp.asarray(y)
    params, opt, loss0 = step(params, opt, x_j, y_j)
    print(f"loss_step1 {float(loss0):.6f}")
    for i in range(4):
        params, opt, loss = step(params, opt, x_j, y_j)
        print(f"loss_step{i + 2} {float(loss):.6f}")

    ctx = test[: cfg.context]
    gen = np.asarray(greedy_generate(params, jnp.asarray(ctx), 16, cfg))
    text = decode_tokens(np.concatenate([np.asarray(ctx), gen]))
    print(f"sample {text!r}")
    print(f"token_acc {token_accuracy(gen, test[cfg.context : cfg.context + 16]):.3f}")
    print(f"grammar {grammar_score(np.concatenate([np.asarray(ctx), gen]), 'deterministic'):.3f}")


if __name__ == "__main__":
    main()
