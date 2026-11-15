import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coop_agents.losses import (
    combined_loss,
    confidence,
    cross_entropy,
    entropy,
    group_loss,
    isolated_loss,
    softmax_weights,
)
from coop_agents.models import mlp, softmax


def test_isolated_ce_one_hot_and_uniform():
    labels = jnp.array([1.0, 0.0])
    correct = jnp.array([1.0, 0.0])
    uniform = jnp.array([0.5, 0.5])
    assert float(isolated_loss(correct, labels)) == pytest.approx(0.0, abs=1e-6)
    assert float(isolated_loss(uniform, labels)) == pytest.approx(np.log(2.0), rel=1e-5)
    assert float(cross_entropy(correct, labels)) == pytest.approx(0.0, abs=1e-6)
    assert float(cross_entropy(uniform, labels)) == pytest.approx(np.log(2.0), rel=1e-5)


def test_confidence_one_hot_is_one_uniform_is_zero():
    one_hot = jnp.array([1.0, 0.0, 0.0])
    uniform = jnp.ones((3,)) / 3.0
    assert float(confidence(one_hot)) == pytest.approx(1.0, abs=1e-5)
    assert float(confidence(uniform)) == pytest.approx(0.0, abs=1e-5)


def test_eq5_weights_in_unit_interval():
    probs = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
        ]
    )
    w = confidence(probs)
    assert jnp.all(w >= -1e-8)
    assert jnp.all(w <= 1.0 + 1e-8)
    # Unnormalized: sum is not required to be 1.
    assert not jnp.isclose(jnp.sum(w), 1.0)
    w_hat = w / jnp.sum(w)
    assert jnp.isclose(jnp.sum(w_hat), 1.0)
    # Paper eq. (5) already sums to 1.
    w5 = softmax_weights(probs, tau=1.0)
    assert jnp.all(w5 >= 0.0)
    assert jnp.isclose(jnp.sum(w5), 1.0)


def test_combined_loss_lambda_zero_is_isolated_squared():
    isol = jnp.array(0.7)
    group = jnp.array(1.3)
    got = combined_loss(isol, group, 0.0)
    assert jnp.isclose(got, isol * isol)
    assert not jnp.isclose(got, isol)
    assert jnp.isclose(combined_loss(isol, group, 1.0), isol * (isol + group))


def test_group_loss_is_ce_of_consensus():
    pred = jnp.array([0.7, 0.3])
    cons = jnp.array([1.0, 0.0])
    assert jnp.isclose(group_loss(pred, cons), isolated_loss(pred, cons))


def test_entropy_of_one_hot_is_zero():
    assert float(entropy(jnp.array([0.0, 1.0]))) == pytest.approx(0.0, abs=1e-6)


def test_softmax_rows_sum_to_one():
    init, apply = mlp([2, 8, 4])
    params = init(jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(2), (16, 2))
    p = softmax(apply(params, x))
    sums = jnp.sum(p, axis=-1)
    assert jnp.allclose(sums, 1.0)
    assert jnp.all(p >= 0.0)
