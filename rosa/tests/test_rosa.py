"""Algorithm 1 checks on a toy stack. No Llama weights are loaded."""

import jax
import jax.numpy as jnp
import numpy as np

from rosa import (
    RosaTrainer,
    active_set,
    adam_moments,
    adam_optimizer_bytes,
    copy_params,
    forward,
    init_affine_stack,
    loss_and_grads,
    mask_count,
    partition_layers,
    rosa_optimizer_bytes,
    select_bottom_rho,
)
from rosa.mask import active_parameter_mask
from rosa.schedule import layers_before


def _equal(left, right):
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _regression(depth=8, dim=4, n=16, seed=0):
    key = jax.random.PRNGKey(seed)
    k_params, k_x, k_y = jax.random.split(key, 3)
    params = init_affine_stack(depth, dim, k_params)
    x = jax.random.normal(k_x, (n, dim), dtype=jnp.float32)
    y = 0.1 * jax.random.normal(k_y, (n, dim), dtype=jnp.float32)
    return params, x, y


def _toy_schedule():
    return partition_layers(list(range(1, 9)), frozen=(1, 2), anchor=(8,), group_size=2)


def test_toy_schedule_matches_algorithm_1():
    schedule = _toy_schedule()
    assert schedule.K == 3
    assert schedule.K > 1
    assert list(schedule.rotatable) == [3, 4, 5, 6, 7]
    assert schedule.groups == ((3, 4), (5, 6), (7,))
    assert len(schedule.groups[-1]) < schedule.group_size
    assert schedule.frozen == frozenset({1, 2})
    assert schedule.anchor == frozenset({8})

    expected = {
        1: (1, (3, 4), frozenset({3, 4, 8}), 3),
        2: (2, (5, 6), frozenset({5, 6, 8}), 5),
        3: (3, (7,), frozenset({7, 8}), 7),
        4: (1, (3, 4), frozenset({3, 4, 8}), 3),
    }
    for epoch, (k, group, active, l_start) in expected.items():
        plan = active_set(schedule, epoch)
        assert plan.k == 1 + ((epoch - 1) % schedule.K)
        assert plan.k == k
        assert plan.group == group
        assert plan.active == active
        assert plan.l_start == l_start
        assert plan.l_start == min(group)


def test_llama_shaped_partition_without_weights():
    layers = ["embedding", *range(32), "head"]
    frozen = ["embedding", *range(8)]
    anchor = [31, "head"]
    schedule = partition_layers(layers, frozen, anchor, group_size=8)

    assert schedule.frozen == frozenset(frozen)
    assert schedule.anchor == frozenset(anchor)
    assert list(schedule.rotatable) == list(range(8, 31))
    assert schedule.groups == (
        tuple(range(8, 16)),
        tuple(range(16, 24)),
        tuple(range(24, 31)),
    )
    assert schedule.K == 3
    assert schedule.K > 1
    assert len(schedule.groups[-1]) == 7
    assert len(schedule.groups[-1]) < 8

    epoch_groups = {
        1: (1, tuple(range(8, 16)), 8),
        2: (2, tuple(range(16, 24)), 16),
        3: (3, tuple(range(24, 31)), 24),
        4: (1, tuple(range(8, 16)), 8),
    }
    anchor_set = frozenset({31, "head"})
    for epoch, (k, group, l_start) in epoch_groups.items():
        plan = active_set(schedule, epoch)
        assert plan.k == k
        assert plan.group == group
        assert plan.l_start == l_start
        assert plan.active == anchor_set | frozenset(group)


def test_bottom_rho_mask_on_known_weights():
    weight = np.array([[4.0, -0.1], [0.7, 0.05], [-2.0, 0.1]], dtype=np.float64)
    # |w| = 4, 0.1, 0.7, 0.05, 2, 0.1. rho=0.5 keeps 3 of 6.
    # Smallest are index 3 (0.05), then index 1 (0.1), then index 5 (0.1).
    mask = select_bottom_rho(weight, 0.5)
    assert mask_count(6, 0.5) == 3
    assert mask.shape == weight.shape
    assert mask.reshape(-1).tolist() == [False, True, False, True, False, True]

    signed = np.array([-10.0, 0.3, 5.0, -0.1])
    signed_mask = select_bottom_rho(signed, 0.5)
    assert signed_mask.tolist() == [False, True, False, True]

    largest = np.array([100.0, 0.01, 50.0])
    assert select_bottom_rho(largest, 1.0 / 3.0).tolist() == [False, True, False]


def test_trainer_mask_is_bottom_rho_of_active_set():
    params, _, _ = _regression(dim=2, seed=1)
    for layer_id, layer in params.items():
        params[layer_id] = {
            "w": jnp.full_like(layer["w"], 3.0),
            "b": jnp.full_like(layer["b"], 3.0),
        }
    params[3]["w"] = params[3]["w"].at[0, 0].set(jnp.float32(1e-4))
    params[3]["b"] = params[3]["b"].at[1].set(jnp.float32(-1e-5))

    schedule = _toy_schedule()
    plan = active_set(schedule, 1)
    per_layer = int(np.asarray(params[1]["w"]).size + np.asarray(params[1]["b"]).size)
    rho = 2 / (per_layer * len(plan.active))
    trainer = RosaTrainer(params, schedule, rho=rho, delta_t=600)
    trainer.begin_epoch(1)

    expected = frozenset({(3, "w", 0), (3, "b", 1)})
    assert trainer.mask == expected
    assert frozenset(active_parameter_mask(trainer.params, trainer.active, rho)) == expected
    assert trainer.active == frozenset({3, 4, 8})


def test_frozen_and_inactive_rotatable_layers_stay_bitwise_identical():
    params, x, y = _regression(seed=2)
    schedule = _toy_schedule()
    trainer = RosaTrainer(params, schedule, rho=1.0, delta_t=50, learning_rate=1e-2)
    before = copy_params(trainer.params)

    trainer.begin_epoch(1)
    for _ in range(5):
        trainer.step(x, y)

    for layer_id in (1, 2, 5, 6, 7):
        assert _equal(before[layer_id]["w"], trainer.params[layer_id]["w"])
        assert _equal(before[layer_id]["b"], trainer.params[layer_id]["b"])
    changed = any(
        not _equal(before[layer_id]["w"], trainer.params[layer_id]["w"])
        for layer_id in (3, 4, 8)
    )
    assert changed

    at_boundary = copy_params(trainer.params)
    trainer.begin_epoch(2)
    for _ in range(4):
        trainer.step(x, y)
    for layer_id in (1, 2, 3, 4, 7):
        assert _equal(at_boundary[layer_id]["w"], trainer.params[layer_id]["w"])
        assert _equal(at_boundary[layer_id]["b"], trainer.params[layer_id]["b"])


def test_truncated_backward_stops_below_l_start():
    params, x, y = _regression(seed=3)
    schedule = _toy_schedule()
    layer_ids = list(schedule.layers)

    plan = active_set(schedule, 2)
    assert plan.l_start == 5
    _, grads = loss_and_grads(params, x, y, layer_ids, plan.l_start)
    for layer_id in layers_before(layer_ids, plan.l_start):
        assert jnp.all(grads[layer_id]["w"] == 0)
        assert jnp.all(grads[layer_id]["b"] == 0)
    assert jnp.any(grads[5]["w"] != 0)
    # Layer 7 sits between G_2 and the anchor. It is not updated, but the
    # backward path still runs through it so G_k can be reached.
    assert jnp.any(grads[7]["w"] != 0)
    assert 7 not in plan.active

    plan = active_set(schedule, 3)
    assert plan.l_start == 7
    _, deeper = loss_and_grads(params, x, y, layer_ids, plan.l_start)
    for layer_id in layers_before(layer_ids, plan.l_start):
        assert jnp.all(deeper[layer_id]["w"] == 0)
        assert jnp.all(deeper[layer_id]["b"] == 0)
    assert jnp.any(deeper[7]["w"] != 0)
    assert len(layers_before(layer_ids, 7)) > len(layers_before(layer_ids, 5))


def test_full_forward_still_uses_the_prefix():
    params, x, _ = _regression(seed=4)
    schedule = _toy_schedule()
    plan = active_set(schedule, 2)
    first = forward(params, x, schedule.layers, plan.l_start)
    edited = copy_params(params)
    edited[1]["w"] = jnp.zeros_like(edited[1]["w"])
    second = forward(edited, x, schedule.layers, plan.l_start)
    assert not jnp.allclose(first, second)


def test_optimizer_state_exists_only_for_current_mask():
    params, x, y = _regression(dim=3, seed=5)
    schedule = _toy_schedule()
    trainer = RosaTrainer(params, schedule, rho=0.2, delta_t=600)
    trainer.begin_epoch(1)
    trainer.step(x, y)

    assert set(trainer.opt_states) == set(trainer.mask)
    assert len(trainer.opt_states) == mask_count(
        sum(np.asarray(trainer.params[lid]["w"]).size + np.asarray(trainer.params[lid]["b"]).size for lid in trainer.active),
        0.2,
    )
    for slot in trainer.opt_states:
        assert slot[0] in trainer.active
        assert slot in trainer.mask
    outside = [
        slot
        for slot in trainer.opt_states
        if slot[0] not in trainer.active
    ]
    assert outside == []


def test_crossing_a_group_boundary_cold_starts_moments():
    params, x, y = _regression(seed=6)
    schedule = _toy_schedule()
    trainer = RosaTrainer(params, schedule, rho=1.0, delta_t=100, learning_rate=5e-2)
    trainer.begin_epoch(1)
    for _ in range(4):
        trainer.step(x, y)

    group_one = [slot for slot in trainer.opt_states if slot[0] in (3, 4)]
    assert group_one
    saved = {}
    for slot in group_one:
        mu, nu, count = adam_moments(trainer.opt_states[slot])
        saved[slot] = np.array(mu)
        assert int(np.asarray(count)) >= 1
    assert any(np.max(np.abs(mu)) > 0 for mu in saved.values())

    trainer.begin_epoch(2)
    for slot in saved:
        assert slot not in trainer.opt_states
    new_group = [slot for slot in trainer.opt_states if slot[0] in (5, 6)]
    assert new_group
    for slot in new_group:
        mu, nu, count = adam_moments(trainer.opt_states[slot])
        assert float(np.asarray(mu)) == 0.0
        assert float(np.asarray(nu)) == 0.0
        assert int(np.asarray(count)) == 0

    for _ in range(2):
        trainer.step(x, y)
    trainer.begin_epoch(3)
    for _ in range(2):
        trainer.step(x, y)
    trainer.begin_epoch(4)
    assert trainer.active == frozenset({3, 4, 8})
    assert trainer.l_start == 3
    restored = False
    for slot, old_mu in saved.items():
        assert slot in trainer.opt_states
        mu, nu, count = adam_moments(trainer.opt_states[slot])
        assert float(np.asarray(mu)) == 0.0
        assert float(np.asarray(nu)) == 0.0
        assert int(np.asarray(count)) == 0
        if np.max(np.abs(old_mu)) > 0:
            assert not np.array_equal(np.asarray(mu), old_mu)
            restored = True
    assert restored


def test_mask_refreshes_every_delta_t_steps():
    params, x, y = _regression(dim=2, seed=7)
    schedule = _toy_schedule()
    params[3]["w"] = params[3]["w"].at[0, 0].set(jnp.float32(50.0))
    trainer = RosaTrainer(params, schedule, rho=0.25, delta_t=2, learning_rate=1e-3)
    trainer.begin_epoch(1)
    watched = (3, "w", 0)
    assert watched not in trainer.mask
    first_mask = set(trainer.mask)
    trainer.step(x, y)
    assert set(trainer.mask) == first_mask
    trainer.step(x, y)
    assert trainer.t == 2
    assert watched not in trainer.mask
    for layer_id in trainer.active:
        trainer.params[layer_id]["w"] = jnp.full_like(trainer.params[layer_id]["w"], 2.0)
        trainer.params[layer_id]["b"] = jnp.full_like(trainer.params[layer_id]["b"], 2.0)
    trainer.params[3]["w"] = trainer.params[3]["w"].at[0, 0].set(jnp.float32(1e-8))
    trainer.step(x, y)
    assert watched in trainer.mask
    assert set(trainer.opt_states) == set(trainer.mask)


def test_memory_scaling_identity():
    psi = 100.0
    k_opt = 4.0
    active_layers = 2.0
    depth = 4.0
    rho = 0.5
    adam = adam_optimizer_bytes(psi, k_opt)
    rosa = rosa_optimizer_bytes(psi, k_opt, active_layers, depth, rho)
    assert adam == 400.0
    assert rosa == 100.0
    assert rosa == adam * (active_layers / depth) * rho

    schedule = _toy_schedule()
    plan = active_set(schedule, 1)
    l = len(plan.active)
    scaled = rosa_optimizer_bytes(80, 8, l, schedule.depth, 0.2)
    assert scaled == (l / schedule.depth) * 80 * 0.2 * 8
    assert l == 3
    assert schedule.depth == 8


def test_short_synthetic_train_returns_finite_loss():
    params, x, y = _regression(seed=8)
    schedule = _toy_schedule()
    trainer = RosaTrainer(params, schedule, rho=0.2, delta_t=2, learning_rate=1e-2)
    losses = []
    for epoch in range(1, 5):
        plan = trainer.begin_epoch(epoch)
        assert plan.l_start == min(plan.group)
        for _ in range(3):
            losses.append(trainer.step(x, y))
    assert len(losses) == 12
    assert all(np.isfinite(loss) for loss in losses)
    assert trainer.plan.k == 1
    assert trainer.active == frozenset({3, 4, 8})
