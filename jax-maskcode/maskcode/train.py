"""Training loop, loss, and BLER/BER evaluation. Section IV-D and Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import bp
from .channel import ChannelConfig
from .codes import LinearBlockCode
from .model import MaskCode, run_batch


def sample_codewords(
    code: LinearBlockCode, batch: int, rng: np.random.Generator
) -> np.ndarray:
    """Lines 7-8 of Algorithm 1: draw m uniformly, then c = mG over GF(2)."""
    m = rng.integers(0, 2, size=(batch, code.k)).astype(np.uint8)
    return code.encode(m)


def bce_loss(c_hat_bp: jax.Array, c: jax.Array) -> jax.Array:
    """Equation (36), summed over the n bits and averaged over the batch.

        -E[ sum_i c_i log sig(c_hat_BP,i) + (1 - c_i) log(1 - sig(c_hat_BP,i)) ]

    optax.sigmoid_binary_cross_entropy is the numerically stable softplus form
    of the same expression; it takes logits, and c_hat_BP is a vector of LLRs,
    which is what a logit for Pr(c_i = 1) is.
    """
    per_bit = optax.sigmoid_binary_cross_entropy(c_hat_bp, c)
    return jnp.mean(jnp.sum(per_bit, axis=-1))


def loss_fn(
    model: MaskCode,
    code: LinearBlockCode,
    c: jax.Array,
    channel: ChannelConfig,
    key: jax.Array,
    n_bp: int,
) -> jax.Array:
    """Lines 19-24 of Algorithm 1, with N_BP unrolled BP iterations in the loop.

    n_bp = 0 is the inner-code-only objective, which is the setting the paper
    ends up recommending.
    """
    c_hat, _ = run_batch(model, code, c, channel, key)
    c_hat_bp = bp.decode(c_hat, code.h, n_bp)
    return bce_loss(c_hat_bp, c)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch: int = 64
    lr: float = 1e-3
    lr_final: float = 3e-7
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    n_bp: int = 0
    seed: int = 0


def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    """AdamW with cosine annealing, as in Sec. V-A.

    The paper anneals 1e-3 -> 3e-7 over 120,101 steps at batch 2,048. Nothing
    here runs for more than a few hundred steps, so the schedule is compressed
    rather than reproduced.

    The global norm clip is not in the paper. The paper instead throws away any
    update whose gradient norm came back infinite, which is what it does to
    survive end-to-end BP training. Clipping is the same idea with a finite
    threshold and keeps the tests from depending on how often that happens.
    """
    schedule = optax.cosine_decay_schedule(
        cfg.lr, cfg.steps, alpha=cfg.lr_final / cfg.lr
    )
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=cfg.weight_decay),
    )


def train(
    model: MaskCode,
    code: LinearBlockCode,
    channel: ChannelConfig,
    cfg: TrainConfig,
) -> tuple[MaskCode, list[float]]:
    """Algorithm 1. Returns the trained model and the per-step loss."""
    opt = make_optimizer(cfg)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.key(cfg.seed)

    @eqx.filter_jit
    def step(model, opt_state, c, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, code, c, channel, key, cfg.n_bp
        )
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    history = []
    for _ in range(cfg.steps):
        key, sub = jax.random.split(key)
        c = jnp.asarray(sample_codewords(code, cfg.batch, rng), dtype=jnp.float32)
        model, opt_state, loss = step(model, opt_state, c, sub)
        history.append(float(loss))
    return model, history


@dataclass(frozen=True)
class Metrics:
    bler_inner: float
    bler_outer: float
    ber_inner: float
    ber_outer: float


def evaluate(
    model: MaskCode,
    code: LinearBlockCode,
    channel: ChannelConfig,
    n_blocks: int = 2048,
    batch: int = 512,
    n_bp: int = 20,
    seed: int = 12345,
) -> Metrics:
    """BLER and BER before and after the outer BP decoder.

    "inner" is the hard decision on the inner decoder LLRs c_hat, i.e. what the
    feedback code alone achieves. "outer" is after BP(c_hat; n_bp), which is the
    number the paper's BLER curves report. The paper evaluates with 20 BP
    iterations, which it finds is enough (Fig. 9).

    A block error is any codeword with at least one wrong bit. Note this counts
    codeword errors, not message errors; the two coincide since m -> c is
    injective.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    @partial(jax.jit, static_argnums=())
    def _hard(c_hat):
        return bp.hard_decision(c_hat), bp.hard_decision(bp.decode(c_hat, code.h, n_bp))

    tot = 0
    err_bits_in = err_bits_out = err_blk_in = err_blk_out = 0
    while tot < n_blocks:
        size = min(batch, n_blocks - tot)
        c = sample_codewords(code, size, rng)
        key, sub = jax.random.split(key)
        c_hat, _ = run_batch(
            model, code, jnp.asarray(c, dtype=jnp.float32), channel, sub
        )
        hin, hout = _hard(c_hat)
        hin, hout = np.asarray(hin), np.asarray(hout)
        err_bits_in += int((hin != c).sum())
        err_bits_out += int((hout != c).sum())
        err_blk_in += int((hin != c).any(axis=1).sum())
        err_blk_out += int((hout != c).any(axis=1).sum())
        tot += size
    return Metrics(
        bler_inner=err_blk_in / tot,
        bler_outer=err_blk_out / tot,
        ber_inner=err_bits_in / (tot * code.n),
        ber_outer=err_bits_out / (tot * code.n),
    )


def bp_gradient_norm(
    model: MaskCode,
    code: LinearBlockCode,
    channel: ChannelConfig,
    n_bp: int,
    key: jax.Array,
    batch: int = 32,
    seed: int = 0,
) -> float:
    """Global norm of dLoss/dtheta when the loss is taken through N_BP BP iterations.

    This is the quantity Remark 1 is about. Equation (37) writes dc_hat_BP/dc_hat
    as a product of N_BP per-iteration Jacobians, and argues the product blows up
    because arctanh'(x) = 1/(1-x^2) diverges as the BP messages saturate. n_bp = 0
    is the reference: the same loss with no BP in the path at all.
    """
    rng = np.random.default_rng(seed)
    c = jnp.asarray(sample_codewords(code, batch, rng), dtype=jnp.float32)
    grads = eqx.filter_grad(loss_fn)(model, code, c, channel, key, n_bp)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    return float(optax.global_norm(leaves))


def bp_backprop_norm(
    code: LinearBlockCode,
    c: jax.Array,
    c_hat: jax.Array,
    n_bp: int,
    eps: float = 1e-7,
) -> jax.Array:
    """Norm of dLoss/dc_hat when the loss is taken through N_BP unrolled BP iterations.

    This is the left-hand side of Remark 1 with the loss attached. By the chain
    rule the gradient factors through (37),

        dLoss/dc_hat = (dLoss/dc_hat_BP) J^(N_BP) ... J^(1),

    so watching this norm as N_BP grows watches the product of Jacobians grow.
    n_bp = 0 removes BP from the path entirely and gives the reference norm.

    Isolating the BP block like this rather than differentiating the whole
    network keeps the measurement about BP and not about whatever the encoder
    happens to be doing at initialization.
    """

    def obj(x):
        return bce_loss(bp.decode(x, code.h, n_bp, eps), c)

    return optax.global_norm(jax.grad(obj)(c_hat))


def main(argv=None):
    import argparse

    from .channel import ChannelConfig
    from .codes import hamming74
    from .model import MaskCode, MaskCodeConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    code = hamming74()
    cfg = MaskCodeConfig(width=args.width, n_layers=args.layers, n_heads=4, phases=3)
    model = MaskCode(code, cfg, key=jax.random.key(args.seed))
    channel = ChannelConfig.from_snr_db(0.0, 20.0)
    train_cfg = TrainConfig(steps=args.steps, batch=args.batch, seed=args.seed)
    _, history = train(model, code, channel, train_cfg)
    print(f"steps={args.steps} first={history[0]:.4f} last={history[-1]:.4f}")


if __name__ == "__main__":
    main()
