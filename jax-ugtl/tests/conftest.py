import jax
import numpy as np
import pytest

from ugtl.surrogate import train_ensemble
from ugtl.tasks import branin_task


@pytest.fixture(scope="session")
def small_task():
    """A small negated-Branin offline dataset, shared across tests."""
    return branin_task(jax.random.PRNGKey(0), n_samples=2000, keep_fraction=0.6)


@pytest.fixture(scope="session")
def scaled_task(small_task):
    """The same dataset with scores min-max scaled to [0, 1], which is what the
    pipeline works in."""
    y = np.asarray(small_task.y)
    return np.asarray(small_task.X), ((y - y.min()) / (y.max() - y.min())).astype(np.float32)


@pytest.fixture(scope="session")
def surrogate(scaled_task):
    X, y = scaled_task
    return train_ensemble(jax.random.PRNGKey(1), X, y, n_members=4, width=64, steps=400)
