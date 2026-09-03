"""Figure 2's three heads, and the property that tells them apart."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hllm.heads import HEAD_KINDS, HeadConfig, apply_head, count_params, init_head


def toy(kind, n_positions=6, d_model=16, **kw):
    return HeadConfig(kind=kind, d_model=d_model, n_positions=n_positions,
                      width=32, n_layers=2, n_heads=4, **kw)


@pytest.mark.parametrize("kind", HEAD_KINDS)
@pytest.mark.parametrize("n_items,n_positions", [(6, 6), (10, 6), (4, 9), (1, 1)])
def test_emits_an_n_by_k_score_matrix(kind, n_items, n_positions):
    cfg = toy(kind, n_positions=n_positions)
    params = init_head(cfg, jax.random.key(0))
    hidden = jax.random.normal(jax.random.key(1), (n_items, cfg.d_model))
    scores = apply_head(cfg, params, hidden)
    assert scores.shape == (n_items, n_positions)
    assert np.isfinite(np.asarray(scores)).all()


@pytest.mark.parametrize("kind", HEAD_KINDS)
def test_equivariant_to_relabelling_the_items(kind):
    """A slate is a set. Permuting the items has to permute the rows of M and
    do nothing else, which is why no head puts a positional encoding on the
    item axis."""
    cfg = toy(kind)
    params = init_head(cfg, jax.random.key(2))
    hidden = jax.random.normal(jax.random.key(3), (7, cfg.d_model))
    perm = np.array([3, 0, 6, 1, 5, 2, 4])
    a = np.asarray(apply_head(cfg, params, hidden))[perm]
    b = np.asarray(apply_head(cfg, params, hidden[perm]))
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_linear_probe_scores_each_item_independently():
    """The point of head (a): no cross-item comparison happens in the head, so
    perturbing one item cannot move another item's scores."""
    cfg = toy("linear_probe")
    params = init_head(cfg, jax.random.key(4))
    hidden = jax.random.normal(jax.random.key(5), (8, cfg.d_model))
    other = hidden.at[3].set(hidden[3] + 10.0)
    a = np.asarray(apply_head(cfg, params, hidden))
    b = np.asarray(apply_head(cfg, params, other))
    moved = np.abs(a - b).max(axis=1)
    assert moved[3] > 1e-3
    assert np.delete(moved, 3).max() == 0.0


@pytest.mark.parametrize("kind", ["self_attention", "slot_query"])
def test_comparison_heads_do_mix_across_items(kind):
    """The same probe, run against the heads that are supposed to fail it."""
    cfg = toy(kind)
    params = init_head(cfg, jax.random.key(6))
    hidden = jax.random.normal(jax.random.key(7), (8, cfg.d_model))
    other = hidden.at[3].set(hidden[3] + 10.0)
    a = np.asarray(apply_head(cfg, params, hidden))
    b = np.asarray(apply_head(cfg, params, other))
    moved = np.abs(a - b).max(axis=1)
    assert np.delete(moved, 3).max() > 1e-6


def test_linear_probe_parameter_count_matches_table_1():
    """D = 1024, K = 50 as in Section 4.1 and Section 5.1: Linear(D, 512) plus
    Linear(512, K) is 524,800 + 25,650."""
    cfg = HeadConfig(kind="linear_probe", d_model=1024, n_positions=50)
    params = init_head(cfg, jax.random.key(8))
    assert count_params(params) == 550_450


def test_self_attention_encoder_parameter_count_matches_table_1():
    """Table 1 says 4.2M for head (b). The two encoder blocks at width 512 with
    an MLP ratio of 2 account for it; the projections are counted apart."""
    cfg = HeadConfig(kind="self_attention", d_model=1024, n_positions=50)
    params = init_head(cfg, jax.random.key(9))
    blocks = count_params(params["blocks"])
    assert round(blocks / 1e6, 1) == 4.2
    assert count_params(params) == blocks + 550_450 + 1024


def test_slot_query_parameter_count_is_near_table_1():
    """Table 1 says 4.0M for head (c). Two cross-attention blocks at width 512
    give 4.19M, which is the closest reading of Figure 2c; the number does not
    fall out exactly."""
    cfg = HeadConfig(kind="slot_query", d_model=1024, n_positions=50)
    params = init_head(cfg, jax.random.key(10))
    assert 3.9e6 < count_params(params["blocks"]) < 4.3e6


def test_init_is_deterministic_in_the_key():
    cfg = toy("self_attention")
    a = init_head(cfg, jax.random.key(11))
    b = init_head(cfg, jax.random.key(11))
    c = init_head(cfg, jax.random.key(12))
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
    assert not all(
        np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(c))
    )


@pytest.mark.parametrize("kind", HEAD_KINDS)
def test_gradients_reach_every_parameter(kind):
    cfg = toy(kind)
    params = init_head(cfg, jax.random.key(13))
    hidden = jax.random.normal(jax.random.key(14), (6, cfg.d_model))
    grads = jax.grad(lambda p: jnp.sum(apply_head(cfg, p, hidden) ** 2))(params)
    for leaf in jax.tree.leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()
    assert any(np.abs(np.asarray(leaf)).max() > 0 for leaf in jax.tree.leaves(grads))


def test_rejects_bad_shapes_and_kinds():
    cfg = toy("linear_probe")
    params = init_head(cfg, jax.random.key(15))
    with pytest.raises(ValueError):
        apply_head(cfg, params, jnp.zeros((2, 3, cfg.d_model)))
    with pytest.raises(ValueError):
        apply_head(cfg, params, jnp.zeros((4, cfg.d_model + 1)))
    with pytest.raises(ValueError):
        HeadConfig(kind="mlp")
    with pytest.raises(ValueError):
        HeadConfig(width=30, n_heads=4)
