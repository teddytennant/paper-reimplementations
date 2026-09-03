"""Toy flow matching training loop, so f-loss has something to be dropped into.

Nothing in the training step knows which objective it is running. The loss is
picked by the (w_f, w_v) pair of Eq. (4), and the network, the interpolant, the
optimizer and the data are identical either way. That is the paper's "drop-in
replacement, no architectural changes" claim, at toy scale.

Following Table 5: linear interpolant x_t = (1 - t) * eps + t * x_1,
x-prediction, logit-normal time sampling with mu = -0.8 and sigma = 0.8, AdamW
with betas (0.9, 0.95) and zero weight decay, (1 - t) clipped at 0.05 in the
division. Not following Table 5: batch 64 instead of 1024, learning rate 1e-3
instead of 2e-4 with no warmup, a few hundred steps instead of 750k, no EMA, and
16x16 synthetic images instead of ImageNet at 256.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .data import sample_images
from .losses import fv_loss
from .model import Predictor, predict_batch
from .schedule import loss_weights
from .spectral import band_relative_error

MU_T, SIGMA_T = -0.8, 0.8  # Table 5, logit(t) ~ N(mu, sigma^2)

# The band this demo scores. It sits above the planted detail modes' neighbours
# and below Nyquist, and it holds the least energy of any band in the 1/f^2
# background, so it is the last thing a spectrally biased model learns.
EVAL_BAND = (0.25, 1.0 / 3.0)


def sample_t(key, batch, dtype=jnp.float32):
    """Logit-normal timesteps, Table 5."""
    z = jax.random.normal(key, (batch,), dtype=dtype)
    return jax.nn.sigmoid(MU_T + SIGMA_T * z)


def interpolate(x1, eps, t):
    """x_t = (1 - t) * eps + t * x_1, the linear interpolant of Section 3.1."""
    tt = t.reshape((-1, 1, 1, 1))
    return (1.0 - tt) * eps + tt * x1


def batch_loss(model, x1, eps, t, w_f, w_v):
    pred = predict_batch(model, interpolate(x1, eps, t), t)
    return fv_loss(pred, x1, t, w_f, w_v)


@eqx.filter_jit
def train_step(model, opt_state, optimizer, x1, eps, t, w_f, w_v):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x1, eps, t, w_f, w_v)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


def run_training(
    seed: int,
    steps: int = 900,
    *,
    switch_step=None,
    switch_width: float = 50.0,
    batch: int = 64,
    size: int = 16,
    learning_rate: float = 1e-3,
    eval_at=(),
    dtype=jnp.float32,
):
    """Train once; return {step: relative error in EVAL_BAND} and the model.

    ``switch_step=None`` runs pure v-loss, w_f = 0 at every step. Any number
    runs the scheduled fv-loss of Eq. (4) with lambda(s) centered there.

    The evaluation set is fixed across seeds and modes: 256 held-out images, one
    fixed noise draw, and t = 0.5, so the only thing that varies between two
    runs is the objective and the training randomness.
    """
    key = jax.random.key(seed)
    key, k_model, k_eval, k_noise = jax.random.split(key, 4)
    model = Predictor(k_model, size=size)
    model = jax.tree.map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, model
    )
    optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.95, weight_decay=0.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    eval_x1 = sample_images(k_eval, 256, size=size, dtype=dtype)
    eval_eps = jax.random.normal(k_noise, eval_x1.shape, dtype=dtype)
    eval_t = jnp.full((256,), 0.5, dtype=dtype)
    eval_xt = interpolate(eval_x1, eval_eps, eval_t)

    zero = jnp.zeros((), dtype)
    one = jnp.ones((), dtype)
    history = {}
    for step in range(steps):
        key, k_data, k_eps, k_t = jax.random.split(key, 4)
        x1 = sample_images(k_data, batch, size=size, dtype=dtype)
        eps = jax.random.normal(k_eps, x1.shape, dtype=dtype)
        t = sample_t(k_t, batch, dtype=dtype)
        if switch_step is None:
            w_f, w_v = zero, one
        else:
            w_f, w_v = loss_weights(step, switch_step, switch_width)
            w_f, w_v = w_f.astype(dtype), w_v.astype(dtype)
        model, opt_state, _ = train_step(
            model, opt_state, optimizer, x1, eps, t, w_f, w_v
        )
        if (step + 1) in eval_at:
            pred = predict_batch(model, eval_xt, eval_t)
            history[step + 1] = float(band_relative_error(pred, eval_x1, *EVAL_BAND))
    return history, model


def main():
    steps = 900
    marks = (300, 600, 900)
    print(f"relative error in band {EVAL_BAND[0]:.3f}-{EVAL_BAND[1]:.3f} cycles/pixel")
    print("seed  step    v-loss   fv-loss")
    for seed in range(3):
        v, _ = run_training(seed, steps=steps, eval_at=marks)
        fv, _ = run_training(seed, steps=steps, switch_step=steps // 2, eval_at=marks)
        for step in marks:
            print(f"{seed:4d}  {step:4d}   {v[step]:7.4f}   {fv[step]:7.4f}")


if __name__ == "__main__":
    main()
