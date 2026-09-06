"""Algorithm 1: rollout, weight, weighted update, refresh every S steps.

Run it with

    python -m osw_ft.train --steps 500

The outer loop draws a batch, spends K inference-only rollouts per query to get
p_hat, stores (q, o*, w) in the buffer M, and then takes `refresh_every` gradient
steps on that batch with the weights frozen. refresh_every = 1 is Algorithm 1 as
printed; larger values are the S of Theorem 2, where holding stale weights costs
the B * eps_p term in eq (10).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .loss import g0_sq_norms, weighted_sft_loss
from .model import ModelConfig, init_params
from .rollout import estimate_weights, success_probability
from .task import ANSWER_LEN, PROMPT_LEN, VOCAB_SIZE, build_dataset
from .weighting import hard_threshold_weight, random_weight, sigma_squared, sigma_squared_sft

__all__ = ["TrainConfig", "train", "compute_weights", "main"]


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 16
    K: int = 2
    temperature: float = 1.0
    lr: float = 3e-3
    refresh_every: int = 1  # the S of Theorem 2
    seed: int = 0
    weight_mode: str = "osw"  # osw | sft | hard | random
    n_digits: int = 10
    eval_every: int = 50
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            vocab_size=VOCAB_SIZE, max_len=PROMPT_LEN + ANSWER_LEN
        )
    )


def compute_weights(params, config, key, batch, K, temperature, mode, running_mean=0.5):
    """Algorithm 1 lines 5-7, or one of the section 4.4 ablation controls."""
    if mode == "sft":
        return jnp.ones((batch.shape[0],))
    rewards_key, noise_key = jr.split(key)
    if mode == "random":
        return random_weight(noise_key, running_mean, (batch.shape[0],))
    from .rollout import rollout_rewards  # noqa: PLC0415

    rewards = rollout_rewards(params, config, rewards_key, batch, PROMPT_LEN, K, temperature)
    if mode == "hard":
        return hard_threshold_weight(rewards)
    if mode == "osw":
        return 1.0 - jnp.mean(rewards, axis=-1)
    raise ValueError(f"unknown weight_mode {mode!r}")


def train(cfg: TrainConfig, verbose=False):
    """Run the outer loop and return a history dict.

    History carries the mean weight, the mean exact success rate p_s, and the
    eq (11) second moment measured against the SFT baseline on the current
    batch, so you can watch the weight fall as the model masters the data.
    """
    key = jr.PRNGKey(cfg.seed)
    key, init_key = jr.split(key)
    params = init_params(init_key, cfg.model)
    data = jnp.asarray(build_dataset(cfg.n_digits))

    optimizer = optax.adamw(cfg.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, batch, w):
        loss, grads = jax.value_and_grad(weighted_sft_loss)(
            params, cfg.model, batch, PROMPT_LEN, w
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = {"step": [], "loss": [], "mean_weight": [], "mean_p_s": [], "sigma2": [], "sigma2_sft": []}
    batch = None
    w = None
    running_mean = 0.5

    for step in range(cfg.steps):
        if step % cfg.refresh_every == 0:
            key, batch_key, weight_key = jr.split(key, 3)
            idx = jr.choice(batch_key, data.shape[0], (cfg.batch_size,), replace=False)
            batch = data[idx]
            w = compute_weights(
                params, cfg.model, weight_key, batch, cfg.K, cfg.temperature,
                cfg.weight_mode, running_mean,
            )
            running_mean = 0.9 * running_mean + 0.1 * float(jnp.mean(w))

        params, opt_state, loss = update(params, opt_state, batch, w)

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps - 1):
            p_s = success_probability(params, cfg.model, batch, PROMPT_LEN, cfg.temperature)
            sq = g0_sq_norms(params, cfg.model, batch, PROMPT_LEN)
            history["step"].append(step)
            history["loss"].append(float(loss))
            history["mean_weight"].append(float(jnp.mean(w)))
            history["mean_p_s"].append(float(jnp.mean(p_s)))
            history["sigma2"].append(float(sigma_squared(p_s, sq, cfg.K)))
            history["sigma2_sft"].append(float(sigma_squared_sft(sq)))
            if verbose:
                print(
                    f"step {step:5d}  loss {float(loss):8.4f}  "
                    f"mean w {float(jnp.mean(w)):.3f}  "
                    f"mean p_s {float(jnp.mean(p_s)):.3f}  "
                    f"sigma2 {history['sigma2'][-1]:.4f} vs sft {history['sigma2_sft'][-1]:.4f}"
                )

    history["params"] = params
    return history


def evaluate(params, cfg: TrainConfig):
    """Exact pass rate over the whole dataset at the sampling temperature."""
    data = jnp.asarray(build_dataset(cfg.n_digits))
    return float(
        jnp.mean(success_probability(params, cfg.model, data, PROMPT_LEN, cfg.temperature))
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Online Self-Weighted Fine-Tuning")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollouts", "-K", type=int, default=2, dest="K")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--refresh-every", type=int, default=1, help="S in Theorem 2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-mode", default="osw", choices=["osw", "sft", "hard", "random"])
    parser.add_argument("--n-digits", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        K=args.K,
        temperature=args.temperature,
        lr=args.lr,
        refresh_every=args.refresh_every,
        seed=args.seed,
        weight_mode=args.weight_mode,
        n_digits=args.n_digits,
        eval_every=args.eval_every,
    )
    history = train(cfg, verbose=True)
    print(f"final exact pass rate over all {cfg.n_digits ** 2} queries: "
          f"{evaluate(history['params'], cfg):.3f}")
    return history


if __name__ == "__main__":
    main()
