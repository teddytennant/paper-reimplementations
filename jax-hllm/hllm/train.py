"""Phase 2 of Section 4.3: fit a scoring head to stored teacher permutations.

Training minimizes equation (3) over the Sinkhorn plan of equation (2).
Evaluation throws Sinkhorn away and decodes with the Hungarian algorithm on M
directly, which is what the last sentence of Section 3.3 says to do and is why
every decoded ranking is a valid permutation.

The defaults here are a toy: 8 items, 32 dimensional item vectors, a head at
width 64. The paper's shapes live in :class:`hllm.heads.HeadConfig`.

    python -m hllm.train --steps 500
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hllm.data import sample_slates, teacher_direction
from hllm.heads import HeadConfig, apply_head, count_params, init_head
from hllm.hungarian import decode_ranking
from hllm.metrics import chance_ndcg, exact_match, ndcg, top1_accuracy
from hllm.sinkhorn import sinkhorn_loss

__all__ = ["TrainConfig", "train", "evaluate", "main"]


@dataclass(frozen=True)
class TrainConfig:
    head: str = "self_attention"
    steps: int = 500
    batch: int = 32
    n_items: int = 8
    d_model: int = 32
    width: int = 64
    n_heads: int = 4
    n_layers: int = 2
    tau: float = 0.5
    sinkhorn_iters: int = 20
    lr: float = 3e-3
    noise: float = 0.1
    eval_slates: int = 256
    seed: int = 0
    log_every: int = 100


def _head_config(cfg: TrainConfig) -> HeadConfig:
    return HeadConfig(
        kind=cfg.head,
        d_model=cfg.d_model,
        n_positions=cfg.n_items,
        width=cfg.width,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
    )


def evaluate(head_cfg: HeadConfig, params, slates):
    """Decode a batch of slates with the Hungarian decoder and score them."""
    scores = jax.vmap(lambda h: apply_head(head_cfg, params, h))(slates.hidden)
    scores = np.asarray(scores)
    decoded = np.stack([decode_ranking(m) for m in scores])
    teacher = np.asarray(slates.order)
    return {
        "ndcg": float(ndcg(decoded, teacher).mean()),
        "ndcg@1": float(ndcg(decoded, teacher, k=1).mean()),
        "recall@1": float(top1_accuracy(decoded, teacher).mean()),
        "exact": float(exact_match(decoded, teacher).mean()),
    }


def train(cfg: TrainConfig = TrainConfig(), verbose: bool = True):
    """Train one head and return ``(params, results)``."""
    head_cfg = _head_config(cfg)
    key = jax.random.key(cfg.seed)
    key, k_init, k_eval = jax.random.split(key, 3)

    params = init_head(head_cfg, k_init)
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)

    # Held-out slates come from the same teacher direction but are never
    # trained on; the split is over slates, not over the label function.
    w = teacher_direction(cfg.d_model, seed=cfg.seed)
    held_out = sample_slates(k_eval, cfg.eval_slates, cfg.n_items, cfg.d_model,
                             noise=cfg.noise, w=w)

    def loss_fn(params, batch):
        def one(hidden, teacher):
            scores = apply_head(head_cfg, params, hidden)
            return sinkhorn_loss(scores, teacher, tau=cfg.tau,
                                 n_iters=cfg.sinkhorn_iters)

        return jnp.mean(jax.vmap(one)(batch.hidden, batch.teacher))

    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    if verbose:
        print(f"head {cfg.head}, {count_params(params):,} params, "
              f"N = K = {cfg.n_items}, D = {cfg.d_model}")

    start = time.perf_counter()
    losses = []
    for i in range(cfg.steps):
        key, k_batch = jax.random.split(key)
        batch = sample_slates(k_batch, cfg.batch, cfg.n_items, cfg.d_model,
                              noise=cfg.noise, w=w)
        params, opt_state, loss = step(params, opt_state, batch)
        losses.append(float(loss))
        if verbose and (i + 1) % cfg.log_every == 0:
            print(f"step {i + 1:5d}  loss {np.mean(losses[-cfg.log_every:]):.4f}")
    elapsed = time.perf_counter() - start

    results = evaluate(head_cfg, params, held_out)
    results["loss"] = float(np.mean(losses[-cfg.log_every:])) if losses else float("nan")
    results["chance_ndcg"] = float(
        chance_ndcg(np.random.default_rng(cfg.seed), np.asarray(held_out.order)).mean()
    )
    results["seconds"] = elapsed
    results["params"] = count_params(params)

    if verbose:
        print(f"\ntrained in {elapsed:.1f}s on {cfg.eval_slates} held-out slates:")
        print(f"  NDCG          {results['ndcg']:.4f}   "
              f"(random ranking: {results['chance_ndcg']:.4f})")
        print(f"  NDCG@1        {results['ndcg@1']:.4f}")
        print(f"  Recall@1      {results['recall@1']:.4f}")
        print(f"  exact perm    {results['exact']:.4f}   "
              f"(chance: {1.0 / math.factorial(cfg.n_items):.2e})")
    return params, results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head", default=TrainConfig.head,
                   choices=["linear_probe", "self_attention", "slot_query"])
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--batch", type=int, default=TrainConfig.batch)
    p.add_argument("--n-items", type=int, default=TrainConfig.n_items)
    p.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    p.add_argument("--width", type=int, default=TrainConfig.width)
    p.add_argument("--n-layers", type=int, default=TrainConfig.n_layers)
    p.add_argument("--tau", type=float, default=TrainConfig.tau)
    p.add_argument("--sinkhorn-iters", type=int, default=TrainConfig.sinkhorn_iters)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = p.parse_args(argv)
    cfg = replace(TrainConfig(), **vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
