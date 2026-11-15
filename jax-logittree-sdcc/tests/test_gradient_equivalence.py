"""The headline claim: the exact walks have the naive tree's gradients.

Proposition 3 says regrouping the leaf sum by branch is an identity, and
Theorem 1 says the packed mask computes the same scalar function of theta. Both
are checked here against ``reference.materialized_tree_loss``, which builds one
sequence per target and never packs anything. Everything runs in float64, and
the comparison covers every leaf of the parameter pytree, not just the loss.
"""

import jax
import jax.numpy as jnp
import pytest

from memorywalker.exact import logittree_loss, packed4d_loss
from memorywalker.reference import materialized_tree_loss

from conftest import make_model, random_rollout, tree_allclose


def test_x64_is_on():
    assert jnp.zeros(1).dtype == jnp.float64


@pytest.mark.parametrize("method", [logittree_loss, packed4d_loss])
def test_matches_materialized_tree_on_the_running_example(method, example):
    params, cfg = make_model(seed=1)
    ref_loss, ref_grad = jax.value_and_grad(materialized_tree_loss)(params, example, cfg)
    loss, grad = jax.value_and_grad(method)(params, example, cfg)
    assert float(ref_loss) > 0
    assert jnp.allclose(loss, ref_loss, atol=1e-10, rtol=1e-10)
    tree_allclose(grad, ref_grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method", [logittree_loss, packed4d_loss])
def test_matches_materialized_tree_on_random_schedules(method, seed):
    params, cfg = make_model(seed=seed, n_layers=2)
    rollout = random_rollout(seed + 10, length=13, n_evictions=3)
    assert rollout.n_junctions >= 2
    assert rollout.diverging_leaves()
    ref_loss, ref_grad = jax.value_and_grad(materialized_tree_loss)(params, rollout, cfg)
    loss, grad = jax.value_and_grad(method)(params, rollout, cfg)
    assert jnp.allclose(loss, ref_loss, atol=1e-10, rtol=1e-10)
    tree_allclose(grad, ref_grad, atol=1e-9, rtol=1e-9)


def test_holds_at_depth_four(example):
    """Nothing about the equivalence should depend on how deep the model is."""
    params, cfg = make_model(seed=3, n_layers=4, d_model=16, n_heads=2, d_ff=32)
    ref_loss, ref_grad = jax.value_and_grad(materialized_tree_loss)(params, example, cfg)
    for method in (logittree_loss, packed4d_loss):
        loss, grad = jax.value_and_grad(method)(params, example, cfg)
        assert jnp.allclose(loss, ref_loss, atol=1e-10, rtol=1e-10)
        tree_allclose(grad, ref_grad, atol=1e-9, rtol=1e-9)


def test_the_two_exact_walks_agree_with_each_other(example):
    """Theorem 1, stated between the two materializations rather than against
    the reference: same loss, same gradient, up to floating-point association."""
    params, cfg = make_model(seed=7, n_layers=3)
    a_loss, a_grad = jax.value_and_grad(logittree_loss)(params, example, cfg)
    b_loss, b_grad = jax.value_and_grad(packed4d_loss)(params, example, cfg)
    assert jnp.allclose(a_loss, b_loss, atol=1e-11, rtol=1e-11)
    tree_allclose(a_grad, b_grad, atol=1e-10, rtol=1e-10)


def test_the_pitfalls_do_not_pass_this_test(example):
    """A guard on the guard: if the naive flattenings also matched, the
    equivalence tests above would be measuring nothing."""
    from memorywalker.naive import naive_compressed_loss, naive_full_loss
    params, cfg = make_model(seed=1)
    ref = materialized_tree_loss(params, example, cfg)
    a = naive_compressed_loss(params, example, cfg)
    b = naive_full_loss(params, example, cfg)
    assert not jnp.allclose(a, ref, atol=1e-3)
    assert not jnp.allclose(b, ref, atol=1e-3)


def test_trunk_tokens_are_scored_once(example):
    """The per-branch loss mask has to cover the response exactly once. If the
    replicated trunk were scored on every branch, the loss would be inflated by
    a whole multiple of the trunk's contribution."""
    from memorywalker.exact import branch_readout
    rows = [p for k in range(example.n_junctions + 1) for p in branch_readout(example, k)[2]]
    assert sorted(rows) == list(example.response_positions())
