"""Where Eq. 7's physical-union mask stops reproducing the tree.

Appendix G says the union view is "the row-wise characterization of what the
block-packed execution computes". That holds for a one-layer model, where every
key is an embedding and so is branch-independent. It stops holding at two
layers: a trunk token has one hidden state in the union, but the tree wants it
to have a different one on each branch, since its own prefix differs there. The
branch-replicated packing is what fixes it, which is why ``exact.py`` uses that.
"""

import jax
import jax.numpy as jnp

from memorywalker.exact import logittree_loss
from memorywalker.tree import Eviction, Rollout, running_example
from memorywalker.union_view import (logical_lifetime, union_is_representable,
                                     union_loss, union_mask)

from conftest import make_model, tree_allclose


def test_lifetimes_match_the_running_example():
    lif = logical_lifetime(running_example())
    assert lif[5] == (5, 8)     # evicted by J_1 = 7, so live for rows up to 7
    assert lif[8] == (8, 11)
    assert lif[11] == (11, 13)
    assert lif[6][1] == float("inf")


def test_running_example_is_representable():
    assert union_is_representable(running_example())


def test_a_schedule_with_no_union_mask():
    """Two targets share a predictor row and disagree about what was live, so no
    single-row mask over H_T can serve both."""
    r = Rollout(tokens=tuple(range(9)), evictions=(Eviction(junction=6, span=(2, 6)),))
    assert r.live_view(6) != r.live_view(7)
    assert max(r.live_view(6)) == max(r.live_view(7)) == 5
    assert not union_is_representable(r)
    try:
        union_mask(r)
    except ValueError:
        pass
    else:
        raise AssertionError("expected union_mask to refuse this schedule")


def test_union_mask_is_exact_at_one_layer():
    example = running_example()
    params, cfg = make_model(seed=4, n_layers=1)
    a, ga = jax.value_and_grad(union_loss)(params, example, cfg)
    b, gb = jax.value_and_grad(logittree_loss)(params, example, cfg)
    assert jnp.allclose(a, b, atol=1e-10, rtol=1e-10)
    tree_allclose(ga, gb, atol=1e-9, rtol=1e-9)


def test_union_mask_drifts_at_two_layers():
    example = running_example()
    params, cfg = make_model(seed=4, n_layers=2)
    a = union_loss(params, example, cfg)
    b = logittree_loss(params, example, cfg)
    assert not jnp.allclose(a, b, atol=1e-4)


def test_union_mask_row_visibility():
    """Row 7 reads out y_8, which was decoded after E_1 fired, so it must not
    admit y_5, even though row 7's own live view still contains it."""
    example = running_example()
    _, bias, _, rows, _ = union_mask(example)
    assert 7 in [int(r) for r in rows]
    assert bias[7, 5] < 0          # closed for the target it predicts
    assert bias[6, 5] == 0.0       # row 6 reads out y_7, which did see y_5
