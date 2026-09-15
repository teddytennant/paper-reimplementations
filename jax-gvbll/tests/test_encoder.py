"""Inductive GraphSAGE encoder tests."""

import jax
import jax.numpy as jnp

from gvbll.encoder import encode, init_graphsage, to_dense_adj


def test_encoder_inductive_different_n():
    key = jax.random.PRNGKey(2)
    params = init_graphsage(key, in_dim=5, hidden_dim=8, out_dim=4)
    n1, n2 = 9, 14
    x1 = jax.random.normal(jax.random.PRNGKey(3), (n1, 5))
    x2 = jax.random.normal(jax.random.PRNGKey(4), (n2, 5))
    adj1 = (jax.random.uniform(jax.random.PRNGKey(5), (n1, n1)) > 0.7).astype(jnp.float32)
    adj1 = jnp.maximum(adj1, adj1.T)
    adj2 = (jax.random.uniform(jax.random.PRNGKey(6), (n2, n2)) > 0.7).astype(jnp.float32)
    adj2 = jnp.maximum(adj2, adj2.T)
    h1 = encode(params, x1, adj1)
    h2 = encode(params, x2, adj2)
    assert h1.shape == (n1, 4)
    assert h2.shape == (n2, 4)


def test_encoder_accepts_coo():
    key = jax.random.PRNGKey(7)
    params = init_graphsage(key, in_dim=3, hidden_dim=6, out_dim=5)
    n = 8
    x = jax.random.normal(jax.random.PRNGKey(8), (n, 3))
    rows = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    cols = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    coo = jnp.stack([rows, cols], axis=0)
    h_coo = encode(params, x, coo)
    dense = to_dense_adj(coo, n)
    h_dense = encode(params, x, dense)
    assert h_coo.shape == (n, 5)
    assert jnp.allclose(h_coo, h_dense, atol=1e-5)
