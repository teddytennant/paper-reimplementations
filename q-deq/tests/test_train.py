"""W8A8 re-forward is used in the training step; loss is finite."""

import jax
import jax.numpy as jnp

from q_deq.deq import head, init_head, init_shared_params
from q_deq.quantize import quantized_shared_layer
from q_deq.solver import qdeq_solve, training_forward


def test_training_step_uses_w8a8_reforward_and_finite_loss():
    hidden, input_dim, out_dim = 8, 12, 3
    key = jax.random.PRNGKey(13)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    layer = init_shared_params(k1, hidden=hidden, input_dim=input_dim, mlp_hidden=8)
    hd = init_head(k2, hidden, out_dim)
    x = jax.random.normal(k3, (input_dim,))
    y = jax.random.normal(k4, (out_dim,))
    z0 = jnp.zeros((hidden,))

    def F(z, xx):
        from q_deq.deq import shared_layer

        return shared_layer(layer, z, xx)

    z_star = qdeq_solve(F, z0, x, k5, k_inner=2, n_dirs=2, n_sweeps=20, n_restarts=1)
    y_q = training_forward(
        layer, hd, x, k5, z0=z0, k_inner=2, n_dirs=2, n_sweeps=20, use_w8a8=True
    )
    y_manual = head(hd, quantized_shared_layer(layer, jax.lax.stop_gradient(z_star), x))
    assert jnp.allclose(y_q, y_manual, atol=1e-4, rtol=1e-4)

    y_fp = training_forward(
        layer, hd, x, k5, z0=z0, k_inner=2, n_dirs=2, n_sweeps=20, use_w8a8=False
    )
    # Random weights are not already on the W8A8 grid.
    assert jnp.linalg.norm(y_q - y_fp) > 1e-8

    def loss_fn(layer_p, head_p):
        pred = training_forward(
            layer_p, head_p, x, k5, z0=z0, k_inner=2, n_dirs=2, n_sweeps=20
        )
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(layer, hd)
    assert jnp.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
