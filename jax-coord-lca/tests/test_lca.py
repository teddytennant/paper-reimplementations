import jax
import jax.numpy as jnp

from coord_lca.lca import (
    bilinear_neighborhood,
    bilinear_sample_tokens,
    cross_attend,
    init_lca,
    localized_cross_attention,
)


def _hand_bilinear(x_down, y_down, height, width):
    x_c = x_down - 0.5
    y_c = y_down - 0.5
    j0 = int(jnp.floor(x_c))
    i0 = int(jnp.floor(y_c))
    wx = float(x_c - j0)
    wy = float(y_c - i0)
    j1, i1 = j0 + 1, i0 + 1
    j0c, j1c = max(0, min(j0, width - 1)), max(0, min(j1, width - 1))
    i0c, i1c = max(0, min(i0, height - 1)), max(0, min(i1, height - 1))
    indices = jnp.array(
        [[i0c, j0c], [i0c, j1c], [i1c, j0c], [i1c, j1c]], dtype=jnp.int32
    )
    weights = jnp.array(
        [(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy],
        dtype=jnp.float32,
    )
    return indices, weights


def test_bilinear_at_token_center_concentrates():
    he, we = 4, 5
    # Center of token (i=2, j=3) is (x_down, y_down) = (3.5, 2.5)
    xy = jnp.array([3.5, 2.5])
    indices, weights, shifted = bilinear_neighborhood(
        xy, encoder_height=he, encoder_width=we
    )
    assert jnp.allclose(shifted, jnp.array([3.0, 2.0]))
    # Weights concentrate on token (2, 3)
    assert float(weights[0]) == 1.0
    assert jnp.allclose(weights[1:], 0.0)
    assert int(indices[0, 0]) == 2 and int(indices[0, 1]) == 3


def test_bilinear_matches_hand_computed():
    he, we = 6, 6
    xy = jnp.array([[1.25, 2.75], [0.0, 0.0], [5.9, 4.1]])
    indices, weights, _ = bilinear_neighborhood(
        xy, encoder_height=he, encoder_width=we
    )
    for n in range(xy.shape[0]):
        want_idx, want_w = _hand_bilinear(float(xy[n, 0]), float(xy[n, 1]), he, we)
        assert jnp.array_equal(indices[n], want_idx)
        assert jnp.allclose(weights[n], want_w, atol=1e-6)
        assert jnp.allclose(jnp.sum(weights[n]), 1.0, atol=1e-5)


def test_lca_output_shape_independent_of_hw():
    key = jax.random.PRNGKey(0)
    d_model = 16
    n_heads = 4
    params = init_lca(key, d_model=d_model, n_heads=n_heads)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (8, 10, d_model))
    n_queries = 7
    query = jax.random.normal(jax.random.PRNGKey(2), (n_queries, d_model))
    xy_enc = jax.random.uniform(jax.random.PRNGKey(3), (n_queries, 2))
    xy_enc = xy_enc * jnp.array([10.0, 8.0])
    out = localized_cross_attention(params, query, tokens, xy_enc)
    assert out.shape == (n_queries, d_model)
    out_noglob = localized_cross_attention(
        params, query, tokens, xy_enc, include_global=False
    )
    assert out_noglob.shape == (n_queries, d_model)


def test_cross_attend_kv_length():
    key = jax.random.PRNGKey(4)
    d_model = 8
    params = init_lca(key, d_model=d_model, n_heads=2)
    query = jax.random.normal(jax.random.PRNGKey(5), (3, d_model))
    kv4 = jax.random.normal(jax.random.PRNGKey(6), (3, 4, d_model))
    kv5 = jax.random.normal(jax.random.PRNGKey(7), (3, 5, d_model))
    assert cross_attend(params, query, kv4).shape == (3, d_model)
    assert cross_attend(params, query, kv5).shape == (3, d_model)


def test_bilinear_sample_tokens_shape():
    tokens = jnp.arange(3 * 4 * 2, dtype=jnp.float32).reshape(3, 4, 2)
    xy = jnp.array([[1.5, 1.5]])
    neighborhood, weights, indices = bilinear_sample_tokens(tokens, xy)
    assert neighborhood.shape == (1, 4, 2)
    assert weights.shape == (1, 4)
    assert indices.shape == (1, 4, 2)
    # Center of token (1, 1) -> first of 2x2 is that token
    assert jnp.allclose(neighborhood[0, 0], tokens[1, 1])
