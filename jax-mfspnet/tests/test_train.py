"""The proxy dataset (Sec. 4.4) and the short training runs that feed Eq. 8."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mfspnet.data import DOWNSCALE_FACTOR, downscale, synthetic_task
from mfspnet.estimators import vle_ema
from mfspnet.model import StackConfig
from mfspnet.train import train_candidate


def test_paper_downscale_factor():
    assert DOWNSCALE_FACTOR == 4  # Sec. 4.4, Table 2


def test_downscale_is_average_pooling_by_n():
    x = jnp.arange(2 * 8 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 8, 3)
    small = downscale(x, 4)
    assert small.shape == (2, 2, 2, 3)
    assert jnp.allclose(small[0, 0, 0], x[0, :4, :4].mean(axis=(0, 1)))
    with pytest.raises(ValueError):
        downscale(x, 3)


def test_synthetic_task_shapes_and_class_balance():
    data = synthetic_task(jax.random.key(0), n_train=64, n_val=32, size=8, n_classes=4)
    assert data.x_train.shape == (64, 8, 8, 3)
    assert data.x_val.shape == (32, 8, 8, 3)
    assert data.n_classes == 4
    assert data.in_channels == 3
    assert len(np.unique(np.asarray(data.y_train))) == 4


def test_training_reduces_the_loss_and_beats_chance():
    data = synthetic_task(jax.random.key(0), n_train=192, n_val=96)
    cfg = StackConfig(n_classes=4, stem_channels=8, n_blocks=1)
    traj = train_candidate(jax.random.key(1), [4, 6], data, cfg, epochs=8, batch_size=64)
    assert traj.train_losses[-1] < traj.train_losses[0]
    assert traj.val_losses[-1] < traj.val_losses[0]
    assert traj.val_accuracy > 0.25  # 4 classes


def test_vle_ema_of_a_real_run_is_a_finite_positive_number():
    data = synthetic_task(jax.random.key(2), n_train=128, n_val=64)
    cfg = StackConfig(n_classes=4, stem_channels=6, n_blocks=1)
    traj = train_candidate(jax.random.key(3), [4], data, cfg, epochs=4, batch_size=64)
    score = vle_ema(traj.val_losses)
    assert np.isfinite(score) and score > 0
    assert traj.val_losses.shape == (4,)
    assert traj.train_losses.shape == (4,)
