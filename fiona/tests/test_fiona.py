"""Tests for packing-aware ternary routing and linear-operator compilation."""

import jax
import jax.numpy as jnp

from fiona import (
    assignment_weights,
    classify_groups,
    compile_routed_linear,
    estimate_saving,
    forward_weights,
    group_sensitivity,
    homogeneity_loss,
    protection_mask,
    soft_ternary_probs,
    ternary_quantize,
    training_objective,
)


def test_ternary_candidates_are_clip_round():
    weights = jnp.array(
        [
            [1.0, -0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.1, -0.2, 0.4],
        ]
    )
    q, gamma, w_hat = ternary_quantize(weights)
    values = {int(v) for v in jnp.unique(q).tolist()}
    assert values <= {-1, 0, 1}
    assert jnp.allclose(w_hat, gamma[:, None] * q)

    # gamma = 0.5, u = [2, -1, 0, 1], clip(round(.)) = [1, -1, 0, 1]
    assert jnp.allclose(gamma[0], 0.5)
    assert jnp.array_equal(q[0], jnp.array([1.0, -1.0, 0.0, 1.0]))
    assert jnp.allclose(w_hat[0], 0.5 * q[0])

    assert float(gamma[1]) == 0.0
    assert jnp.array_equal(q[1], jnp.zeros((4,)))
    assert jnp.array_equal(w_hat[1], jnp.zeros((4,)))

    # round would leave the ternary range before the clip
    assert int(q[2, 0]) == 1


def test_pure_group_has_lower_homogeneity_loss_than_mixed():
    pure = jnp.ones((1, 4))
    mixed = jnp.array([[1.0, -1.0, 0.0, 0.2]])
    loss_pure = homogeneity_loss(soft_ternary_probs(pure), group_size=4)[0, 0]
    loss_mixed = homogeneity_loss(soft_ternary_probs(mixed), group_size=4)[0, 0]
    assert float(loss_pure) < float(loss_mixed)


def test_protection_keeps_ten_highest_sensitivity_pure_groups():
    # 100 groups, 90 pure. rho_p_max = 0.2 limits the signed fraction to 0.8.
    is_pure = jnp.array([True] * 90 + [False] * 10)
    sensitivity = jnp.concatenate(
        [
            jnp.arange(90, dtype=jnp.float32),
            jnp.arange(1000, 1010, dtype=jnp.float32),
        ]
    )
    protected = protection_mask(is_pure, sensitivity, rho_p_max=0.2)
    chosen = [int(i) for i in jnp.where(protected)[0].tolist()]
    assert chosen == list(range(80, 90))
    assert int(jnp.sum(protected)) == 10
    assert not bool(jnp.any(protected & ~is_pure))

    under = protection_mask(jnp.array([True] * 80 + [False] * 20), sensitivity, 0.2)
    assert int(jnp.sum(under)) == 0


def test_paper_example_compiles_and_matches_naive():
    # A1..A4, y1, y2.
    # g11, g21: h=+1; g13, g23: h=-1; g24: h=0 (skip).
    # Remaining groups are raw.
    signed_h = jnp.array(
        [
            [1, 0, -1, 0],
            [1, 0, -1, 0],
        ]
    )
    raw_mask = jnp.array(
        [
            [False, True, False, True],
            [False, True, False, False],
        ]
    )
    plan = compile_routed_linear(signed_h, raw_mask)
    assert plan.naive_pmult == 7
    assert plan.compiled_pmult == 5
    assert plan.compiled_rescale < plan.naive_rescale
    # k=2, m=2 saving is 0, so the heuristic does not accept this share.
    assert estimate_saving(2, 2) == 0
    assert plan.shared_sum_saving == 0
    assert len(plan.shares) == 0

    inputs = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    gammas = jnp.array([[0.5], [0.25]])
    raw_weights = jnp.zeros((2, 4, 1))
    raw_weights = raw_weights.at[0, 1].set(0.1)
    raw_weights = raw_weights.at[0, 3].set(0.2)
    raw_weights = raw_weights.at[1, 1].set(0.3)
    raw_weights = raw_weights.at[1, 3].set(999.0)

    naive = plan.naive(inputs, gammas, raw_weights)
    compiled = plan.compiled(inputs, gammas, raw_weights)
    # y1 = gamma1*A1 - gamma1*A3 + g12*A2 + g14*A4
    # y2 = gamma2*A1 - gamma2*A3 + g22*A2
    expected = jnp.array(
        [
            [0.5 * 1.0 - 0.5 * 3.0 + 0.1 * 2.0 + 0.2 * 4.0],
            [0.25 * 1.0 - 0.25 * 3.0 + 0.3 * 2.0],
        ]
    )
    assert jnp.allclose(naive, expected)
    assert jnp.allclose(compiled, naive)

    key = jax.random.PRNGKey(0)
    packed = jax.random.normal(key, (4, 8))
    gamma_vec = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    raw_vec = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 8))
    assert jnp.allclose(
        plan.naive(packed, gamma_vec, raw_vec),
        plan.compiled(packed, gamma_vec, raw_vec),
        rtol=1e-5,
        atol=1e-5,
    )


def test_overlapping_shared_sums_accept_one_per_output():
    # Two m=3 sums, each shared by k=2 outputs, overlap on output 0 input 2.
    signed_h = jnp.zeros((3, 5), dtype=jnp.int32)
    signed_h = signed_h.at[0, 0:5].set(1)
    signed_h = signed_h.at[1, 0:3].set(1)
    signed_h = signed_h.at[2, 2:5].set(1)
    raw_mask = jnp.zeros((3, 5), dtype=bool)
    assert estimate_saving(2, 3) > 0
    assert estimate_saving(2, 3) == 2 * 3 - 3 - 2

    plan = compile_routed_linear(signed_h, raw_mask)
    used_by_output = [[] for _ in range(3)]
    for share in plan.shares:
        assert share.saving > 0
        for output, _orient in share.users:
            used_by_output[output].append(set(share.inputs))
    for used in used_by_output:
        for left in range(len(used)):
            for right in range(left + 1, len(used)):
                assert used[left].isdisjoint(used[right])
    on_first = used_by_output[0]
    assert len(on_first) == 1
    assert on_first[0] in ({0, 1, 2}, {2, 3, 4})

    inputs = jax.random.normal(jax.random.PRNGKey(3), (5, 4))
    gammas = jax.random.normal(jax.random.PRNGKey(4), (3, 4))
    raw_weights = jnp.zeros((3, 5, 4))
    assert jnp.allclose(
        plan.naive(inputs, gammas, raw_weights),
        plan.compiled(inputs, gammas, raw_weights),
        rtol=1e-5,
        atol=1e-5,
    )


def test_assignment_weights_and_forward_route():
    is_pure = jnp.array([[True, False, False]])
    sensitivity = jnp.array([[9.0, 1.0, 3.0]])
    weights = assignment_weights(is_pure, sensitivity)
    assert float(weights[0, 0]) == 1.0
    # mixed F = [1, 3], population std 1, z = [-1, 1]
    assert jnp.allclose(weights[0, 1], jax.nn.sigmoid(1.0))
    assert jnp.allclose(weights[0, 2], jax.nn.sigmoid(-1.0))
    assert float(weights[0, 1]) > float(weights[0, 2])

    lone = assignment_weights(jnp.array([False]), jnp.array([4.0]))
    assert jnp.allclose(lone, 0.5)

    raw = jnp.array([[1.0, 1.0, 1.0, -1.0, 0.2]])
    q, gamma, hat = ternary_quantize(raw)
    pure, _h = classify_groups(q, group_size=2)
    assert pure.shape == (1, 2)
    assert bool(pure[0, 0]) and not bool(pure[0, 1])
    guarded = jnp.array([[False, False]])
    routed = forward_weights(raw, hat, pure, guarded, group_size=2)
    assert jnp.allclose(routed[0, :2], hat[0, :2])
    assert jnp.allclose(routed[0, 2:], raw[0, 2:])
    guarded = jnp.array([[True, False]])
    routed = forward_weights(raw, hat, pure, guarded, group_size=2)
    assert jnp.allclose(routed, raw)


def test_sensitivity_and_training_objective():
    weights = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    hat = jnp.array([[1.5, 1.5, 3.0, 0.0]])
    curvature = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    sens = group_sensitivity(weights, hat, curvature, group_size=2)
    assert jnp.allclose(sens, jnp.array([[0.75, 64.0]]))

    loss = training_objective(
        task_loss=jnp.array(1.5),
        a=jnp.array([1.0, 0.5]),
        hom_loss=jnp.array([0.2, 0.4]),
        lambda_group=2.0,
    )
    assert jnp.allclose(loss, 1.9)


def test_reversed_signed_set_reuses_negated_sum():
    signed_h = jnp.array(
        [
            [1, 1, 1, 0],
            [-1, -1, -1, 0],
        ]
    )
    raw_mask = jnp.zeros_like(signed_h, dtype=bool)
    plan = compile_routed_linear(signed_h, raw_mask)
    assert len(plan.shares) == 1
    share = plan.shares[0]
    orients = {output: orient for output, orient in share.users}
    assert set(orients) == {0, 1}
    assert orients[0] == -orients[1]
    assert estimate_saving(2, 3) > 0

    inputs = jax.random.normal(jax.random.PRNGKey(5), (4, 3))
    gammas = jax.random.normal(jax.random.PRNGKey(6), (2, 3))
    raw_weights = jnp.zeros((2, 4, 3))
    assert jnp.allclose(
        plan.naive(inputs, gammas, raw_weights),
        plan.compiled(inputs, gammas, raw_weights),
    )


def _reuses_a1_minus_a3(share) -> bool:
    terms = dict(share.terms)
    users = {output: orient for output, orient in share.users}
    if set(users) != {0, 1} or set(terms) != {0, 2}:
        return False
    for orient in users.values():
        if orient * terms[0] != 1 or orient * terms[2] != -1:
            return False
    return True
