"""f-loss has to be usable wherever v-loss was, and the schedule has to be a switch.

The paper's claim is that this "acts as a drop-in replacement for flow matching
losses" with no architectural change. Concretely: same call signature, same
scalar output, jittable and differentiable the same way, and a training step
that only differs in the (w_f, w_v) pair of Eq. (4).

Also pins down lambda(s): sigmoid decaying from 1 to 0, centered on the
crossover point s*, with w_f + w_v = 1 at every step.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

jax.config.update("jax_enable_x64", True)

from flf.losses import f_loss, fv_loss, v_loss
from flf.model import Predictor, predict_batch, velocity
from flf.schedule import frequency_weight, loss_weights
from flf.train import batch_loss, interpolate, sample_t, train_step


@pytest.fixture
def batch():
    keys = jax.random.split(jax.random.key(2), 3)
    pred = jax.random.normal(keys[0], (4, 8, 8, 2), dtype=jnp.float32)
    target = jax.random.normal(keys[1], (4, 8, 8, 2), dtype=jnp.float32)
    t = sample_t(keys[2], 4, dtype=jnp.float32)
    return pred, target, t


def test_both_losses_return_the_same_kind_of_scalar(batch):
    pred, target, t = batch
    a, b = v_loss(pred, target, t), f_loss(pred, target, t)
    assert a.shape == b.shape == ()
    assert a.dtype == b.dtype == pred.dtype


def test_both_losses_jit_and_differentiate(batch):
    pred, target, t = batch
    for fn in (v_loss, f_loss):
        value = jax.jit(fn)(pred, target, t)
        assert np.isfinite(float(value))
        grad = jax.jit(jax.grad(fn))(pred, target, t)
        assert grad.shape == pred.shape
        assert np.all(np.isfinite(np.asarray(grad)))


def test_fv_loss_endpoints_are_the_two_losses(batch):
    pred, target, t = batch
    assert np.isclose(fv_loss(pred, target, t, 0.0, 1.0), v_loss(pred, target, t))
    assert np.isclose(fv_loss(pred, target, t, 1.0, 0.0), f_loss(pred, target, t))


def test_schedule_decays_from_one_to_zero_through_the_switch_point():
    steps = jnp.arange(0, 1000, dtype=jnp.float64)
    lam = np.asarray(frequency_weight(steps, 500.0, 50.0))
    assert np.all(np.diff(lam) < 0.0)
    assert lam[0] > 0.9999
    assert lam[-1] < 1e-4
    assert frequency_weight(500.0, 500.0, 50.0) == pytest.approx(0.5)
    w_f, w_v = loss_weights(jnp.asarray(320.0), 500.0, 50.0)
    assert float(w_f + w_v) == pytest.approx(1.0)
    assert w_f > w_v  # frequencies first, pixels later, the direction of Table 4


def test_training_step_is_identical_apart_from_the_weights():
    """Same model, same data, same optimizer; only w_f and w_v change."""
    key = jax.random.key(0)
    k_model, k_data, k_eps, k_t = jax.random.split(key, 4)
    model = Predictor(k_model, size=8)
    model = jax.tree.map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model
    )
    x1 = jax.random.normal(k_data, (4, 8, 8, 1), dtype=jnp.float32)
    eps = jax.random.normal(k_eps, x1.shape, dtype=jnp.float32)
    t = sample_t(k_t, 4, dtype=jnp.float32)
    optimizer = optax.adamw(1e-3, b1=0.9, b2=0.95, weight_decay=0.0)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    updated = {}
    for name, (w_f, w_v) in {
        "v": (jnp.float32(0.0), jnp.float32(1.0)),
        "f": (jnp.float32(1.0), jnp.float32(0.0)),
        "fv": (jnp.float32(0.7), jnp.float32(0.3)),
    }.items():
        new_model, _, loss = train_step(model, state, optimizer, x1, eps, t, w_f, w_v)
        assert np.isfinite(float(loss))
        updated[name] = eqx.filter(new_model, eqx.is_inexact_array)

    base = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    for name, tree in updated.items():
        leaves = jax.tree.leaves(tree)
        assert [x.shape for x in leaves] == [x.shape for x in base]
        assert [x.dtype for x in leaves] == [x.dtype for x in base]
        assert not np.allclose(leaves[0], base[0])
    a = jax.tree.leaves(updated["v"])[0]
    b = jax.tree.leaves(updated["f"])[0]
    assert not np.allclose(a, b)


def test_batch_loss_jits_under_both_objectives():
    key = jax.random.key(4)
    k_model, k_data, k_eps, k_t = jax.random.split(key, 4)
    model = Predictor(k_model, size=8)
    model = jax.tree.map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, model
    )
    x1 = jax.random.normal(k_data, (3, 8, 8, 1), dtype=jnp.float32)
    eps = jax.random.normal(k_eps, x1.shape, dtype=jnp.float32)
    t = sample_t(k_t, 3, dtype=jnp.float32)
    fn = eqx.filter_jit(batch_loss)
    for w_f, w_v in ((jnp.float32(0.0), jnp.float32(1.0)), (jnp.float32(1.0), jnp.float32(0.0))):
        assert np.isfinite(float(fn(model, x1, eps, t, w_f, w_v)))


def test_x_prediction_and_velocity_are_consistent():
    """v_theta = (x_theta - x_t) / (1 - t), so a perfect x-prediction gives v*."""
    key = jax.random.key(6)
    k1, k2, k3 = jax.random.split(key, 3)
    x1 = jax.random.normal(k1, (5, 8, 8, 1), dtype=jnp.float64)
    eps = jax.random.normal(k2, x1.shape, dtype=jnp.float64)
    t = jax.random.uniform(k3, (5,), dtype=jnp.float64, maxval=0.9)
    x_t = interpolate(x1, eps, t)
    assert np.allclose(velocity(x1, x_t, t), x1 - eps, rtol=1e-12, atol=1e-12)
