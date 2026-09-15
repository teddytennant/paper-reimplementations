"""Short training run must strictly decrease MAP NLL on a two-community graph."""

import jax
import jax.numpy as jnp

from gvbll.data import two_community_graph
from gvbll.train import evaluate, init_gvbll, train


def test_training_decreases_nll():
    key = jax.random.PRNGKey(11)
    k_data, k_init, k_train = jax.random.split(key, 3)
    x, adj, y = two_community_graph(k_data, n_per=10, feat_dim=6, noise=0.1)
    params = init_gvbll(
        k_init, in_dim=x.shape[1], n_classes=2, hidden_dim=16, embed_dim=8
    )
    before = evaluate(params, x, adj, y)
    params, losses = train(
        params,
        x,
        adj,
        y,
        k_train,
        steps=120,
        lr=2e-2,
        n_samples=6,
        anneal_steps=40,
    )
    after = evaluate(params, x, adj, y)
    assert jnp.isfinite(before["nll"])
    assert jnp.isfinite(after["nll"])
    assert float(after["nll"]) < float(before["nll"])
    assert jnp.all(jnp.isfinite(losses))
