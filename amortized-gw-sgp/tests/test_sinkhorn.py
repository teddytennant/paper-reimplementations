import jax
import jax.numpy as jnp

from gwsgp.sinkhorn import sinkhorn


def test_sinkhorn_row_and_column_sums():
    key = jax.random.PRNGKey(0)
    cost = jax.random.uniform(key, (7, 7))
    plan = sinkhorn(cost, epsilon=0.2, n_iters=40)
    row = jnp.sum(plan, axis=1)
    col = jnp.sum(plan, axis=0)
    assert jnp.allclose(row, 1.0, atol=1e-4)
    assert jnp.allclose(col, 1.0, atol=1e-4)
    assert jnp.all(plan >= 0.0)


def test_sinkhorn_batched_marginals():
    key = jax.random.PRNGKey(1)
    cost = jax.random.uniform(key, (3, 5, 5))
    plan = sinkhorn(cost, epsilon=0.2, n_iters=30)
    row = jnp.sum(plan, axis=-1)
    col = jnp.sum(plan, axis=-2)
    assert jnp.allclose(row, 1.0, atol=1e-4)
    assert jnp.allclose(col, 1.0, atol=1e-4)
