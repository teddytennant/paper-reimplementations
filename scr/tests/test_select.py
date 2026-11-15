import jax.numpy as jnp

from scr.select import augment_indices, select_indices


def test_one_head_basis_query_picks_index_zero():
    q = jnp.array([[1.0, 0.0, 0.0]])
    K = jnp.eye(3)[None, :, :]
    idx = select_indices(q, K, k=1)
    assert int(idx[0]) == 0


def test_two_heads_agreeing_token_beats_a_single_head_peak():
    # head 0 peaks at token 0, head 1 peaks at token 1, both like token 2 more than either peak.
    # Construct so the summed softmax top-1 is the shared token.
    d = 2
    q = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    # token 0 is aligned with head 0 only, token 1 with head 1 only, token 2 with both.
    K = jnp.zeros((2, 3, d))
    K = K.at[0, 0].set(jnp.array([1.0, 0.0]))
    K = K.at[1, 1].set(jnp.array([1.0, 0.0]))
    K = K.at[0, 2].set(jnp.array([1.0, 0.0]))
    K = K.at[1, 2].set(jnp.array([1.0, 0.0]))
    idx = select_indices(q, K, k=1)
    assert int(idx[0]) == 2


def test_augment_adds_sinks_and_local_deduped_sorted():
    out = augment_indices([5, 1], n_cache=10, n_init=2, local_positions=[1, 7, 7, 12])
    assert out == [0, 1, 5, 7]
