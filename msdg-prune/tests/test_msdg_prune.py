"""Hand-computed checks for Algorithm 1. CPU only, no images."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from msdg_prune import (
    CLIP_SALIENCY_WINDOW,
    CLIP_VIT_L_D_S,
    DEFAULT_ELL_0,
    DEFAULT_K,
    DEFAULT_SEED,
    DEFAULT_TAU,
    QWEN25_VL_D_S,
    QWEN_SALIENCY_WINDOW,
    allocate_budgets,
    endpoint_displacement,
    farthest_point_init,
    max_query_cosine,
    msdg_prune,
    spherical_kmeans,
    unit,
    update_direction,
)


def test_unit_scales_by_norm_and_floors_near_zero():
    got = unit(jnp.array([3.0, 4.0]))
    # 3/5 and 4/5. Float division is one ulp off the decimal literal 0.6.
    assert float(got[0]) == pytest.approx(0.6)
    assert float(got[1]) == pytest.approx(0.8)

    # ||[1e-15, 0]|| = 1e-15 < 1e-12, so the floor is the divisor.
    tiny = unit(jnp.array([1e-15, 0.0]))
    assert float(tiny[0]) == 1e-3
    assert float(tiny[1]) == 0.0


def test_update_direction_matches_equation_1():
    h_0 = jnp.array([1.0, 0.0])
    h_d = jnp.array([0.0, 1.0])
    got = update_direction(h_d, h_0)
    # unit(unit([0, 1]) - unit([1, 0])) = unit([-1, 1])
    expected_0 = -math.sqrt(2.0) / 2.0
    expected_1 = math.sqrt(2.0) / 2.0
    assert float(got[0]) == pytest.approx(expected_0)
    assert float(got[1]) == pytest.approx(expected_1)


def test_endpoint_displacement_is_l2_gap_not_path_length():
    h_s = jnp.array([0.0, 0.0])
    h_mid = jnp.array([10.0, 0.0])
    h_e = jnp.array([1.0, 0.0])
    assert float(endpoint_displacement(h_e, h_s)) == 1.0
    path = float(jnp.linalg.norm(h_mid - h_s) + jnp.linalg.norm(h_e - h_mid))
    assert path == 19.0

    # Token 0 walks a short path to a far endpoint. Token 1 walks out and back.
    # Endpoint gaps are 5 and 1. Path lengths are 1 + sqrt(20) and 19.
    states = jnp.zeros((2, 6, 2))
    states = states.at[:, 0, :].set(jnp.array([1.0, 0.0]))
    states = states.at[:, 1, :].set(jnp.array([0.0, 1.0]))
    states = states.at[0, 2, :].set(jnp.array([0.0, 0.0]))
    states = states.at[0, 3, :].set(jnp.array([1.0, 0.0]))
    states = states.at[0, 4, :].set(jnp.array([3.0, 4.0]))
    states = states.at[1, 2, :].set(jnp.array([0.0, 0.0]))
    states = states.at[1, 3, :].set(jnp.array([10.0, 0.0]))
    states = states.at[1, 4, :].set(jnp.array([1.0, 0.0]))
    visual = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    queries = jnp.array([[1.0, 0.0]])
    result = msdg_prune(
        states,
        visual,
        queries,
        budget=1,
        ell_star=4,
        n_groups=1,
        ell_s=2,
        ell_e=4,
        ell_0=0,
        ell_d=1,
        d_s=0,
        tau=50.0,
        seed=0,
    )
    assert result.indices.tolist() == [0]


def test_query_relevance_is_max_cosine_and_score_is_product():
    token = jnp.array([3.0, 4.0])
    queries = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    alpha = max_query_cosine(token, queries)
    # cosines are 0.6 and 0.8, so the max is 0.8. s = 0.8 * 5 = 4.
    assert float(alpha) == 0.8
    assert float(alpha) * 5.0 == 4.0

    # Three tokens, one group, budget 1.
    # token 0: alpha 0.6, u 5, s 3.0 (largest u)
    # token 1: alpha 1.0, u 1, s 1.0 (largest alpha)
    # token 2: alpha 0.8, u 4, s 3.2 (largest product)
    states = jnp.zeros((3, 4, 2))
    states = states.at[:, 0, :].set(jnp.array([1.0, 0.0]))
    states = states.at[:, 1, :].set(jnp.array([0.0, 1.0]))
    states = states.at[0, 3, :].set(jnp.array([5.0, 0.0]))
    states = states.at[1, 3, :].set(jnp.array([1.0, 0.0]))
    states = states.at[2, 3, :].set(jnp.array([4.0, 0.0]))
    visual = jnp.array(
        [
            [3.0, 4.0],
            [1.0, 0.0],
            [4.0, 3.0],
        ]
    )
    result = msdg_prune(
        states,
        visual,
        jnp.array([[1.0, 0.0]]),
        budget=1,
        ell_star=0,
        n_groups=1,
        ell_s=2,
        ell_e=3,
        ell_0=0,
        ell_d=1,
        d_s=0,
        tau=50.0,
        seed=0,
    )
    assert result.indices.tolist() == [2]


def test_sink_filter_keeps_threshold_and_drops_above():
    # ell_star = 0, so the diagnostic state is layer 1. A large value on layer 0
    # must not exclude the token. abs == tau is kept. abs > tau is excluded,
    # including a large negative activation.
    states = jnp.zeros((3, 4, 2))
    states = states.at[0, 0, 0].set(999.0)
    states = states.at[0, 1, 0].set(50.0)
    states = states.at[1, 1, 0].set(-51.0)
    states = states.at[2, 1, 0].set(0.0)
    states = states.at[:, 2, :].set(jnp.array([1.0, 0.0]))
    states = states.at[:, 3, :].set(jnp.array([0.0, 1.0]))
    visual = jnp.ones((3, 2))
    result = msdg_prune(
        states,
        visual,
        jnp.array([[1.0, 0.0]]),
        budget=2,
        ell_star=0,
        n_groups=1,
        ell_s=2,
        ell_e=3,
        ell_0=2,
        ell_d=3,
        d_s=0,
        tau=50.0,
        seed=0,
    )
    assert result.indices.tolist() == [0, 2]
    assert result.group_ids.tolist() == [0, -1, 0]


def test_farthest_point_init_seed_zero_then_farthest_tie():
    # Seed 0 picks the first candidate. Tokens 2 and 3 are tied for farthest.
    # Token 1 is closer, so the second centroid is not merely the next index.
    # The tie keeps the earlier original row.
    directions = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    chosen, _ = farthest_point_init(directions, n_groups=2, seed=0)
    assert chosen.tolist() == [0, 2]


def test_allocator_fractional_tie_prefers_lower_index():
    # floor(5 * p) = [2, 1, 1], R = 1, fractional parts [0.5, 0.5, 0.0].
    # The 0.5 tie goes to the lower index, so budgets are [3, 1, 1].
    proportions = jnp.array([0.5, 0.3, 0.2])
    budgets = allocate_budgets(proportions, [5, 5, 5], budget=5)
    assert budgets.tolist() == [3, 1, 1]


def test_allocator_capacity_moves_capped_leftover():
    # p = [0.5, 0.375, 0.125], sizes = [1, 10, 10], B = 6.
    # B * p = [3, 2.25, 0.75], floor = [3, 2, 0], fractional parts [0, 0.25, 0.75].
    # Group 0 has |C| = 1, so floor 3 is capped and b starts at [1, 2, 0], R = 3.
    # One fractional pass adds one to group 2, then group 1: [1, 3, 1], R = 1.
    # The leftover then follows descending p and lands on group 1.
    # A cycle of the fractional pass would have given that last token to group 2.
    proportions = jnp.array([0.5, 0.375, 0.125])
    budgets = allocate_budgets(proportions, [1, 10, 10], budget=6)
    assert budgets.tolist() == [1, 4, 1]


def _two_group_states(end_gaps):
    """Three tokens. Directions put tokens 0 and 2 in group 0, token 1 in group 1."""
    states = jnp.zeros((3, 4, 2))
    # d = [0, 1] for tokens 0 and 2; d = [1, 0] for token 1.
    states = states.at[0, 0, :].set(jnp.array([0.0, -1.0]))
    states = states.at[0, 1, :].set(jnp.array([0.0, 1.0]))
    states = states.at[2, 0, :].set(jnp.array([0.0, -1.0]))
    states = states.at[2, 1, :].set(jnp.array([0.0, 1.0]))
    states = states.at[1, 0, :].set(jnp.array([-1.0, 0.0]))
    states = states.at[1, 1, :].set(jnp.array([1.0, 0.0]))
    for i, gap in enumerate(end_gaps):
        states = states.at[i, 3, 0].set(float(gap))
    return states


def test_retained_indices_are_sorted_across_groups():
    # Group 0 keeps the later token 2. Group 1 keeps token 1.
    # Concatenating groups would yield [2, 1]. The return is original order.
    states = _two_group_states([0.0, 1.0, 2.0])
    visual = jnp.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    result = msdg_prune(
        states,
        visual,
        jnp.array([[1.0, 0.0]]),
        budget=2,
        ell_star=0,
        n_groups=2,
        ell_s=2,
        ell_e=3,
        ell_0=0,
        ell_d=1,
        d_s=0,
        tau=50.0,
        seed=0,
    )
    assert result.group_ids.tolist() == [0, 1, 0]
    assert result.budgets.tolist() == [1, 1]
    assert result.indices.tolist() == [1, 2]
    assert jnp.allclose(result.tokens, visual[jnp.array([1, 2])])


def test_topk_tie_keeps_earlier_index():
    states = _two_group_states([1.0, 1.0, 1.0])
    visual = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    result = msdg_prune(
        states,
        visual,
        jnp.array([[1.0, 0.0]]),
        budget=2,
        ell_star=0,
        n_groups=2,
        ell_s=2,
        ell_e=3,
        ell_0=0,
        ell_d=1,
        d_s=0,
        tau=50.0,
        seed=0,
    )
    assert result.group_ids.tolist() == [0, 1, 0]
    assert result.budgets.tolist() == [1, 1]
    assert result.indices.tolist() == [0, 1]


def test_empty_groups_take_distinct_directions_in_margin_order():
    # Init picks four centroids. Groups 2 and 3 lose the lower-index tie and
    # are empty. All margins are 1, so original order assigns d_0 then d_1.
    directions = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    labels, centroids = spherical_kmeans(directions, n_groups=4, seed=0)
    assert labels.tolist() == [0, 1, 1, 0]
    assert centroids[2].tolist() == [1.0, 0.0]
    assert centroids[3].tolist() == [0.0, 1.0]


def test_zero_update_directions_raise():
    states = jnp.zeros((1, 4, 2))
    states = states.at[0, 2, :].set(jnp.array([1.0, 0.0]))
    states = states.at[0, 3, :].set(jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="max_i"):
        msdg_prune(
            states,
            jnp.array([[1.0, 0.0]]),
            jnp.array([[1.0, 0.0]]),
            budget=1,
            ell_star=0,
            n_groups=1,
            ell_s=2,
            ell_e=3,
            ell_0=2,
            ell_d=3,
            d_s=0,
            tau=50.0,
        )


def test_default_grouping_endpoints_are_ell0_2_and_ell_d_l_minus_1():
    # Shape (1, 5, 2) means indices 0..4, so L = 4 and default ell_d = 3.
    # Only the pair (2, 3) has a nonzero update direction.
    states = jnp.zeros((1, 5, 2))
    states = states.at[0, :, :].set(jnp.array([1.0, 0.0]))
    states = states.at[0, 0, :].set(jnp.array([0.0, 1.0]))
    states = states.at[0, 3, :].set(jnp.array([0.0, 1.0]))
    result = msdg_prune(
        states,
        jnp.array([[1.0, 0.0]]),
        jnp.array([[1.0, 0.0]]),
        budget=1,
        ell_star=0,
        n_groups=1,
        ell_s=0,
        ell_e=1,
        d_s=0,
        tau=50.0,
    )
    assert result.indices.tolist() == [0]


def test_documented_presets():
    assert DEFAULT_TAU == 50.0
    assert CLIP_VIT_L_D_S == 650
    assert QWEN25_VL_D_S == 849
    assert CLIP_SALIENCY_WINDOW == (14, 19)
    assert QWEN_SALIENCY_WINDOW == (19, 24)
    assert DEFAULT_K == 20
    assert DEFAULT_SEED == 0
    assert DEFAULT_ELL_0 == 2
