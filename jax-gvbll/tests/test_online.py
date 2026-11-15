"""Online Laplace update moves the last-layer posterior toward a new label."""

import jax
import jax.numpy as jnp

from gvbll.online import online_laplace_update, pack_posterior
from gvbll.vbll import predict_map


def test_online_laplace_moves_map_toward_observed_class():
    mean = jnp.array(
        [
            [0.6, 0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
        ]
    )
    precision = jnp.ones((2, 4))
    post = pack_posterior(mean, precision)
    post_train = pack_posterior(mean, precision)
    h = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    y = jnp.array([1], dtype=jnp.int32)

    p_before = predict_map(post["mean"], h)[0]
    assert float(p_before[0]) > float(p_before[1])

    post_new = online_laplace_update(
        post,
        h,
        y,
        post_train,
        lambda_=0.9,
        eta=1.0,
        beta=0.0,
        clip_delta=2.0,
    )
    assert not jnp.allclose(post_new["mean"], post["mean"])
    assert not jnp.allclose(post_new["precision"], post["precision"])

    p_after = predict_map(post_new["mean"], h)[0]
    assert float(p_after[1]) > float(p_before[1])


def test_online_update_with_training_anchor():
    key = jax.random.PRNGKey(21)
    mean = 0.1 * jax.random.normal(key, (3, 5))
    precision = jnp.full((3, 5), 2.0)
    post = pack_posterior(mean, precision)
    post_train = pack_posterior(jnp.zeros((3, 5)), jnp.ones((3, 5)))
    h = jax.random.normal(jax.random.PRNGKey(22), (4, 5))
    y = jnp.array([0, 1, 2, 1], dtype=jnp.int32)
    updated = online_laplace_update(
        post, h, y, post_train, lambda_=0.95, eta=0.5, beta=1.0
    )
    for name in ("mean", "precision", "log_std"):
        assert updated[name].shape == post[name].shape
        assert jnp.all(jnp.isfinite(updated[name]))
    assert jnp.all(updated["precision"] > 0)
