import jax
import jax.numpy as jnp

from scr.attend import per_subchunk_select, scr_attention, selection_calls, sparse_attention


def _identical_chunk(M, L, d_h, n_heads=1):
    length = M * L
    Q = jnp.ones((n_heads, length, d_h))
    K = jnp.ones((n_heads, length, d_h))
    V = jnp.ones((n_heads, length, d_h))
    pos = jnp.arange(length)
    return Q, K, V, pos


def test_identical_subchunks_select_once():
    L = 4
    Q, K, V, pos = _identical_chunk(8, L, 4)
    calls = selection_calls(Q, L=L, theta=0.95, Lmax=8 * L)
    assert calls == 1
    per, _ = per_subchunk_select(Q, K, k=2, L=L)
    assert per == 8


def test_orthogonal_fingerprints_select_per_subchunk():
    L = 2
    M = 4
    length = M * L
    # alternating orthogonal flattened queries
    Q = jnp.zeros((1, length, 2))
    Q = Q.at[:, 0::4, 0].set(1.0)
    Q = Q.at[:, 1::4, 0].set(1.0)
    Q = Q.at[:, 2::4, 1].set(1.0)
    Q = Q.at[:, 3::4, 1].set(1.0)
    calls = selection_calls(Q, L=L, theta=0.95, Lmax=100)
    assert calls == M


def test_causal_query_does_not_see_future_key():
    # query at 0, keys at 0 and 5. V at key 5 is huge. A leak would move the output.
    Q = jnp.ones((1, 1, 2))
    K = jnp.ones((1, 2, 2))
    V = jnp.array([[[0.0, 0.0], [100.0, 100.0]]])
    out = sparse_attention(Q, K, V, query_pos=jnp.array([0]), key_pos=jnp.array([0, 5]))
    assert jnp.allclose(out, 0.0, atol=1e-5)


def test_identical_members_share_the_anchor_selection():
    L = 4
    Q, K, V, pos = _identical_chunk(4, L, 4)
    # make keys distinguishable so selection is not a tie
    K = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4)
    out_calls = selection_calls(Q, L=L, theta=0.95, Lmax=100)
    assert out_calls == 1
    _, chosen = per_subchunk_select(Q, K, k=2, L=L)
    for c in chosen[1:]:
        assert jnp.array_equal(c, chosen[0])


def test_sparse_matches_dense_softmax_on_gathered_keys():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    Q = jax.random.normal(k1, (2, 3, 4))
    K = jax.random.normal(k2, (2, 5, 4))
    V = jax.random.normal(k3, (2, 5, 4))
    query_pos = jnp.array([4, 5, 6])
    key_pos = jnp.array([1, 2, 4, 5, 9])
    got = sparse_attention(Q, K, V, query_pos, key_pos)
    scale = jnp.sqrt(jnp.array(4.0))
    logits = jnp.einsum("hqd,hkd->hqk", Q, K) / scale
    allow = key_pos[None, :] <= query_pos[:, None]
    logits = logits + jnp.where(allow, 0.0, -jnp.inf)[None, :, :]
    w = jax.nn.softmax(logits, axis=-1)
    want = jnp.einsum("hqk,hkd->hqd", w, V)
    assert jnp.allclose(got, want, atol=1e-5)
