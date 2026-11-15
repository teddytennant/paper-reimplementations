"""Run a tiny frozen decoder with HeRo FFN routing."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from hero.config import HeroConfig
from hero.model import forward, init_params, total_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy HeRo decoder")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()

    cfg = HeroConfig(n_layers=args.layers)
    key = jax.random.PRNGKey(0)
    params = init_params(key, cfg)
    tokens = jax.random.randint(
        jax.random.PRNGKey(1), (2, args.tokens), 0, cfg.vocab_size
    )
    logits, aux = forward(params, tokens, cfg, return_aux=True)
    loss, _ = total_loss(params, tokens, cfg)
    mean_gate = jnp.mean(aux["gates"])
    print(
        f"logits {tuple(logits.shape)} gates {tuple(aux['gates'].shape)} "
        f"mean_g {float(mean_gate):.4f} loss {float(loss):.4f}"
    )


if __name__ == "__main__":
    main()
