"""With K = 0 every method has to collapse to ordinary sequence training.

Nothing forks, so the tree is one branch, the compressed walk is the physical
trace, and the diverging-leaf set is empty. This is the degenerate case the
whole construction has to pass through cleanly.
"""

import jax
import jax.numpy as jnp

from memorywalker.exact import logittree_loss, packed4d_loss
from memorywalker.model import cross_entropy, sequence_logits
from memorywalker.naive import naive_compressed_loss, naive_full_loss
from memorywalker.reference import materialized_tree_loss
from memorywalker.sdcc import residual_kl, sdcc_loss
from memorywalker.tree import Rollout
from memorywalker.union_view import union_loss

from conftest import make_model, tree_allclose

ROLLOUT = Rollout(tokens=(2, 7, 1, 9, 4, 4, 0, 6, 3, 5), evictions=(), response_start=1)


def plain_loss(params, rollout, cfg):
    logits = sequence_logits(params, jnp.asarray(rollout.tokens), cfg)
    rows = jnp.arange(rollout.response_start - 1, rollout.length - 1)
    return cross_entropy(logits[rows], jnp.asarray(rollout.tokens[rollout.response_start:]))


def test_every_method_reduces_to_plain_sequence_training():
    params, cfg = make_model(seed=11, n_layers=2)
    ref, gref = jax.value_and_grad(plain_loss)(params, ROLLOUT, cfg)
    methods = [materialized_tree_loss, logittree_loss, packed4d_loss,
               naive_compressed_loss, naive_full_loss, union_loss,
               lambda p, r, c: sdcc_loss(p, r, c, lam=1.0)]
    for method in methods:
        loss, grad = jax.value_and_grad(method)(params, ROLLOUT, cfg)
        assert jnp.allclose(loss, ref, atol=1e-10, rtol=1e-10)
        tree_allclose(grad, gref, atol=1e-9, rtol=1e-9)


def test_nothing_diverges():
    assert ROLLOUT.diverging_leaves() == ()
    params, cfg = make_model(seed=11)
    assert residual_kl(params, ROLLOUT, cfg) == {}
