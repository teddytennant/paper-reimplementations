"""Both flattenings leak, and the leak is visible in the logits.

Pitfall B (Naive-Full) lets a token read a span that was evicted before it was
decoded, which is information the deployed model will never get. Pitfall A
(Naive-Compressed) scores a token without a span that was still on screen when
it was decoded, which is information the model did have. The test perturbs one
evicted token and watches which readouts move.
"""

import jax
import jax.numpy as jnp
import pytest

from memorywalker.exact import branch_logits, branch_readout, logittree_loss, packed4d_loss
from memorywalker.model import cross_entropy, sequence_logits
from memorywalker.naive import naive_compressed_loss, naive_full_loss
from memorywalker.reference import materialized_tree_loss, per_token_logits
from memorywalker.tree import Eviction, Rollout

from conftest import make_model

# E_1 = {5} at J_1 = 7. Target 6 was decoded while 5 was visible; target 9 was
# decoded after 5 had gone.
ROLLOUT = Rollout(tokens=(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8),
                  evictions=(Eviction(junction=7, span=(5,)),),
                  response_start=1)
PERTURBED = Rollout(tokens=ROLLOUT.tokens[:5] + (17,) + ROLLOUT.tokens[6:],
                    evictions=ROLLOUT.evictions, response_start=ROLLOUT.response_start)


def _naive_full_logits(params, rollout, cfg, t):
    return sequence_logits(params, jnp.asarray(rollout.tokens), cfg)[t - 1]


def _naive_compressed_logits(params, rollout, cfg, t):
    walk = rollout.compressed_walk()
    ids = jnp.asarray([rollout.tokens[i] for i in walk])
    return sequence_logits(params, ids, cfg)[walk.index(t) - 1]


def _logittree_logits(params, rollout, cfg, t):
    k = rollout.layer_of(t)
    rows, _, phys = branch_readout(rollout, k)
    return branch_logits(params, rollout, k, cfg)[rows[phys.index(t)]]


def _moved(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def test_setup_is_the_case_we_claim():
    assert 5 in ROLLOUT.live_view(6) and 5 not in ROLLOUT.live_view(9)
    assert 5 not in ROLLOUT.compressed_prefix(6)
    assert ROLLOUT.diverging_leaves() == (6, 7)


def test_naive_full_reads_a_span_that_was_already_evicted():
    """Stale-context leakage: y_9's prediction should not depend on y_5, and
    under Naive-Full it does."""
    params, cfg = make_model(seed=2)
    exact = [_logittree_logits, lambda p, r, c, t: per_token_logits(p, r, c)[t]]
    for fn in exact:
        assert _moved(fn(params, ROLLOUT, cfg, 9), fn(params, PERTURBED, cfg, 9)) < 1e-12
    leak = _moved(_naive_full_logits(params, ROLLOUT, cfg, 9),
                  _naive_full_logits(params, PERTURBED, cfg, 9))
    assert leak > 1e-3


def test_naive_compressed_drops_a_span_the_model_had_seen():
    """Time-travel leakage, the mirror: y_6's prediction must depend on y_5,
    and under Naive-Compressed it cannot, because y_5 is not in H_comp at all."""
    params, cfg = make_model(seed=2)
    for fn in (_logittree_logits, lambda p, r, c, t: per_token_logits(p, r, c)[t]):
        assert _moved(fn(params, ROLLOUT, cfg, 6), fn(params, PERTURBED, cfg, 6)) > 1e-3
    frozen = _moved(_naive_compressed_logits(params, ROLLOUT, cfg, 6),
                    _naive_compressed_logits(params, PERTURBED, cfg, 6))
    assert frozen < 1e-12


def _logdiff(params, rollout, cfg, method_logits):
    """Eq. 1: mean |log P(y_t | c_train) - log P(y_t | c_rollout)| over the response.

    The rollout-time term is the materialized tree, since that is by definition
    the context each token was decoded from.
    """
    rollout_logits = per_token_logits(params, rollout, cfg)
    alive = set(rollout.compressed_walk())
    diffs = []
    for t in rollout.response_positions():
        if t not in alive:
            continue
        y = rollout.tokens[t]
        train = jax.nn.log_softmax(method_logits(params, rollout, cfg, t))[y]
        roll = jax.nn.log_softmax(rollout_logits[t])[y]
        diffs.append(abs(float(train - roll)))
    return sum(diffs) / len(diffs)


def test_logdiff_has_a_calibrated_zero():
    """Section 3.1: the drift statistic sits at zero for a conditioning-consistent
    pipeline and lifts off it for both pitfalls."""
    params, cfg = make_model(seed=2)
    assert _logdiff(params, ROLLOUT, cfg, _logittree_logits) < 1e-12
    assert _logdiff(params, ROLLOUT, cfg, _naive_compressed_logits) > 1e-2
    assert _logdiff(params, ROLLOUT, cfg, _naive_full_logits) > 1e-2


@pytest.mark.parametrize("naive", [naive_compressed_loss, naive_full_loss])
def test_naive_losses_are_biased(naive):
    params, cfg = make_model(seed=5)
    ref = materialized_tree_loss(params, ROLLOUT, cfg)
    for exact in (logittree_loss, packed4d_loss):
        assert jnp.allclose(exact(params, ROLLOUT, cfg), ref, atol=1e-10)
    assert abs(float(naive(params, ROLLOUT, cfg) - ref)) > 1e-3


def test_naive_compressed_never_scores_the_evicted_token():
    """It cannot: y_5 has no position in H_comp. That missing term is part of
    why its loss differs from the tree's, alongside the leakage."""
    params, cfg = make_model(seed=5)
    walk = ROLLOUT.compressed_walk()
    assert 5 not in walk
    scored = [i for i in walk if i >= ROLLOUT.response_start and walk.index(i) > 0]
    assert len(scored) == len(ROLLOUT.response_positions()) - 1
    ids = jnp.asarray([ROLLOUT.tokens[i] for i in walk])
    logits = sequence_logits(params, ids, cfg)
    rows = jnp.asarray([walk.index(i) - 1 for i in scored])
    targets = jnp.asarray([ROLLOUT.tokens[i] for i in scored])
    assert jnp.allclose(cross_entropy(logits[rows], targets),
                        naive_compressed_loss(params, ROLLOUT, cfg))
