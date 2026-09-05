"""Encoder sign constraint, power constraint, and a short training run."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maskcode import bp, codes
from maskcode.channel import ChannelConfig, average_power, normalize_power
from maskcode.model import MaskCode, MaskCodeConfig, run_batch
from maskcode.train import TrainConfig, bce_loss, loss_fn, train


def test_initial_encoder_keeps_the_bpsk_sign():
    """Equation (16): x_1 = |o_1| * (2c - 1)."""
    code = codes.hamming74()
    cfg = MaskCodeConfig(width=8, n_layers=1, n_heads=2, phases=2)
    model = MaskCode(code, cfg, key=jax.random.key(0))
    c = jnp.asarray(code.encode(np.array([[1, 0, 1, 0]], dtype=np.uint8)), dtype=jnp.float32)
    bipolar = 2.0 * c - 1.0
    x1 = jax.vmap(model.initial, in_axes=(0, None))(bipolar, code.zero_mask())
    assert np.all(np.asarray(x1) * np.asarray(bipolar) >= 0.0)
    assert np.all(np.sign(np.asarray(x1)) == np.sign(np.asarray(bipolar)))


def test_normalize_power_meets_equation_9():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(32, 7)))
    y = normalize_power(x, 1.0)
    assert abs(float(average_power(y)) - 1.0) < 1e-6
    # a positive scale cannot flip signs
    assert np.all(np.sign(np.asarray(y)) == np.sign(np.asarray(x)))


def test_run_batch_transmissions_meet_the_power_constraint():
    code = codes.hamming74()
    cfg = MaskCodeConfig(width=8, n_layers=1, n_heads=2, phases=3)
    model = MaskCode(code, cfg, key=jax.random.key(1))
    channel = ChannelConfig.from_snr_db(0.0, 20.0)
    c = jnp.asarray(code.codewords()[:16], dtype=jnp.float32)
    _, xs = run_batch(model, code, c, channel, jax.random.key(2))
    for t in range(cfg.phases):
        assert abs(float(average_power(xs[t])) - channel.power) < 1e-5


def test_training_decreases_bce_on_a_fixed_batch():
    """n_bp = 0, tiny Hamming, a few Adam steps on one frozen batch."""
    code = codes.hamming74()
    cfg = MaskCodeConfig(width=8, n_layers=1, n_heads=2, phases=2)
    model = MaskCode(code, cfg, key=jax.random.key(3))
    channel = ChannelConfig.from_snr_db(2.0, 20.0)
    c = jnp.asarray(code.codewords()[:16], dtype=jnp.float32)
    key = jax.random.key(4)
    opt = optax.adam(3e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            model, code, c, channel, key, 0
        )
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    losses = []
    for i in range(20):
        # Same codewords and same channel draw every step, so a drop is the
        # optimizer, not a quieter noise sample.
        model, opt_state, loss = step(model, opt_state, key)
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    assert min(losses) < losses[0]


def test_train_entrypoint_runs():
    code = codes.single_parity_check(4)
    cfg = MaskCodeConfig(width=8, n_layers=1, n_heads=2, phases=2)
    model = MaskCode(code, cfg, key=jax.random.key(5))
    channel = ChannelConfig.from_snr_db(0.0, 20.0)
    _, history = train(
        model, code, channel, TrainConfig(steps=3, batch=8, seed=0)
    )
    assert len(history) == 3
    assert all(np.isfinite(h) for h in history)
