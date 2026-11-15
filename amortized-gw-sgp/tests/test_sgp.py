import jax
import jax.numpy as jnp

from gwsgp.train import init_params, sgp_loss_one, train


def test_amortized_training_decreases_gw_loss():
    key = jax.random.PRNGKey(0)
    _params, loss_before, loss_after = train(
        key,
        n_steps=200,
        batch_size=16,
        n_nodes=6,
        lr=1e-2,
        tau=0.1,
        n_sinkhorn=20,
        omega_weight=1.0,
        eval_size=24,
    )
    assert jnp.isfinite(loss_before)
    assert jnp.isfinite(loss_after)
    assert loss_after < loss_before


def test_sgp_loss_gradients_finite():
    key = jax.random.PRNGKey(4)
    params = init_params(key, n_nodes=6)
    x = jax.nn.one_hot(jnp.array(1), 3)
    adj = jnp.zeros((6, 6)).at[jnp.arange(5), jnp.arange(1, 6)].set(1.0)
    adj = adj + adj.T

    def loss_fn(p):
        total, _aux = sgp_loss_one(p, x, adj, tau=0.15, n_sinkhorn=12)
        return total

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree.leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
