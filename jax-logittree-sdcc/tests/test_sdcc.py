"""SDCC: the reconstruction, the Pinsker bound, and what training does to it.

Proposition 2 says the residual KL at a diverging leaf controls a behavioral
quantity: ``||pi(.|p_t) - pi(.|p_s)||_TV <= sqrt(eps_p / 2)``. That is an
inequality about two distributions the code already computes, so it can be
checked directly rather than argued about, and the O(sqrt(eps_KL)) rate of
Section 5.2 is what it gives once eps is being driven down.
"""

import functools
import math

import jax
import jax.numpy as jnp
import optax
import pytest

from memorywalker.naive import naive_compressed_loss
from memorywalker.sdcc import (leaf_gap, residual_kl, sdcc_loss, teacher_logits,
                               teacher_logits_by_reconstruction, teacher_prefix)
from memorywalker.tree import running_example

from conftest import make_model, random_rollout, tree_allclose


def test_equation_9_rebuilds_the_live_prefix():
    """Trunk, re-inserted span, post-junction suffix. For y_6 the reconstruction
    has to put y_5 back at its original slot."""
    r = running_example()
    assert teacher_prefix(r, 6) == (0, 1, 2, 3, 4, 5)
    assert teacher_prefix(r, 7) == (0, 1, 2, 3, 4, 5, 6)
    assert teacher_prefix(r, 9) == (0, 1, 2, 3, 4, 6, 7, 8)
    assert teacher_prefix(r, 12) == (0, 1, 2, 3, 4, 6, 7, 9, 10, 11)


def test_overlapping_evictions_reinsert_every_active_span():
    """App. D.1: when several evictions are active at p the teacher re-inserts
    all of them, and the result is still the live view."""
    from memorywalker.tree import Eviction, Rollout
    r = Rollout(tokens=tuple(range(12)),
                evictions=(Eviction(junction=8, span=(3,)), Eviction(junction=9, span=(5,))))
    assert len(r.active_evictions(6)) == 2
    assert teacher_prefix(r, 6) == (0, 1, 2, 3, 4, 5)


def test_the_two_teacher_routes_agree():
    """Grouping leaves by branch layer costs at most K no-grad forwards; doing
    one forward per leaf off Eq. 9 costs |C|. They must give the same teachers."""
    r = running_example()
    params, cfg = make_model(seed=6)
    a = teacher_logits(params, r, cfg)
    b = teacher_logits_by_reconstruction(params, r, cfg)
    assert set(a) == set(b) == set(r.diverging_leaves())
    for p in a:
        assert jnp.allclose(a[p], b[p], atol=1e-10)


def test_teacher_forwards_are_at_most_K():
    r = running_example()
    layers = {r.layer_of(p) for p in r.diverging_leaves()}
    assert len(layers) <= r.n_junctions
    assert r.n_junctions not in layers  # the last layer re-converges


def test_lambda_zero_is_naive_compressed():
    r = running_example()
    params, cfg = make_model(seed=6)
    a, ga = jax.value_and_grad(functools.partial(sdcc_loss, lam=0.0))(params, r, cfg)
    b, gb = jax.value_and_grad(naive_compressed_loss)(params, r, cfg)
    assert jnp.allclose(a, b, atol=1e-12, rtol=1e-12)
    tree_allclose(ga, gb, atol=1e-12, rtol=1e-12)


def test_teacher_carries_no_gradient():
    """The teacher is the current policy with a stop-gradient, so freezing it at
    a detached copy of theta cannot change the gradient of the SDCC loss."""
    r = running_example()
    params, cfg = make_model(seed=6)
    g = jax.grad(functools.partial(sdcc_loss, lam=1.0))(params, r, cfg)

    frozen = jax.tree.map(lambda x: x + 0.0, params)
    def loss_with_external_teacher(p):
        task = naive_compressed_loss(p, r, cfg)
        from memorywalker.model import sequence_logits
        from memorywalker.sdcc import forward_kl, student_rows
        rows, leaves = student_rows(r)
        walk = r.compressed_walk()
        student = sequence_logits(p, jnp.asarray([r.tokens[i] for i in walk]), cfg)
        teachers = teacher_logits(frozen, r, cfg)
        return task + sum(forward_kl(teachers[q], student[row]) for row, q in zip(rows, leaves))

    tree_allclose(g, jax.grad(loss_with_external_teacher)(params), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rollout_seed", [21, 22, 23])
def test_pinsker_bound_holds(seed, rollout_seed):
    """Proposition 2, over randomized trials. The bound has to hold at every
    diverging leaf, and be doing real work rather than comparing zero to zero."""
    rollout = random_rollout(rollout_seed, length=12, n_evictions=2)
    params, cfg = make_model(seed=seed, n_layers=2, d_model=16, n_heads=2, d_ff=32)
    gaps = leaf_gap(params, rollout, cfg)
    assert gaps
    saw_positive = False
    for kl, tv in gaps.values():
        kl, tv = float(kl), float(tv)
        assert kl >= -1e-12
        assert tv <= math.sqrt(max(kl, 0.0) / 2) + 1e-12
        saw_positive |= tv > 1e-3
    assert saw_positive


def test_sdcc_training_drives_the_residual_down():
    """The KL residual, and with it the TV gap the bound controls, should shrink
    under the SDCC objective on a toy rollout."""
    rollout = running_example()
    params, cfg = make_model(seed=9, n_layers=2, d_model=16, n_heads=2, d_ff=32)
    objective = functools.partial(sdcc_loss, rollout=rollout, cfg=cfg, lam=10.0)
    loss_fn = jax.jit(objective)
    grad_fn = jax.jit(jax.grad(objective))
    report = jax.jit(functools.partial(leaf_gap, rollout=rollout, cfg=cfg))

    def summary(p):
        gaps = report(p)
        kls = [float(k) for k, _ in gaps.values()]
        tvs = [float(t) for _, t in gaps.values()]
        return sum(kls) / len(kls), max(tvs)

    kl0, tv0 = summary(params)
    opt = optax.adam(3e-2)
    state = opt.init(params)
    trace = []
    for step in range(500):
        grads = grad_fn(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        gaps = report(params)
        for kl, tv in gaps.values():
            # the bound is not just an end-state claim; it holds all the way down
            assert float(tv) <= math.sqrt(max(float(kl), 0.0) / 2) + 1e-12
        if step % 100 == 99:
            trace.append(summary(params)[0])

    kl1, tv1 = summary(params)
    assert kl0 > 1e-2
    assert trace[0] < kl0          # moving within the first hundred steps
    assert kl1 < 0.25 * kl0
    assert tv1 < 0.5 * tv0
    assert float(loss_fn(params)) > 0


def test_residual_kl_is_only_at_diverging_leaves():
    r = running_example()
    params, cfg = make_model(seed=6)
    assert set(residual_kl(params, r, cfg)) == {6, 7, 9, 10, 12}
