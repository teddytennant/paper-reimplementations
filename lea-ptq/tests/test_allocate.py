"""Hand checks for greedy bit allocation."""

from lea_ptq.allocate import greedy_allocate


SCORES = [
    {7: 1.0, 5: 1.4, 3: 3.0},
    {7: 2.0, 5: 2.1, 3: 2.2},
    {7: 0.5, 5: 1.5, 3: 4.0},
]
COUNTS = [4, 2, 8]


def test_drops_minimum_delta_block_first_and_respects_budget():
    # Start [7, 7, 7], cost = 7*4 + 7*2 + 7*8 = 98.
    # Drop 7->5 deltas:
    #   g0: 0.4 / (2*4) = 0.05
    #   g1: 0.1 / (2*2) = 0.025
    #   g2: 1.0 / (2*8) = 0.0625
    # First drop is block 1. Cost becomes 94. Budget 94 stops there.
    assert greedy_allocate(COUNTS, SCORES, budget=94) == [7, 5, 7]


def test_already_within_budget_keeps_max_assignment():
    assert greedy_allocate(COUNTS, SCORES, budget=98) == [7, 7, 7]
    assert greedy_allocate(COUNTS, SCORES, budget=1000) == [7, 7, 7]


def test_tie_breaks_by_smallest_index():
    scores = [
        {7: 1.0, 5: 1.2, 3: 5.0},
        {7: 1.0, 5: 1.2, 3: 5.0},
        {7: 1.0, 5: 2.0, 3: 3.0},
    ]
    # Equal first-step delta on blocks 0 and 1. One drop: cost 210 -> 190.
    assert greedy_allocate([10, 10, 10], scores, budget=190) == [5, 7, 7]


def test_stops_at_floor_candidate():
    # Minimum cost is 3*(4+2+8) = 42. Budget 0 cannot be met.
    assert greedy_allocate(COUNTS, SCORES, budget=0) == [3, 3, 3]
    # After [5, 3, 5] cost is 66. Next min delta drops block 2 to 3, cost 50.
    assert greedy_allocate(COUNTS, SCORES, budget=50) == [5, 3, 3]


def test_budget_is_average_scales_by_element_count():
    # sum(n_g) = 14. Average budget 7 means Bud = 98, so no drop.
    # The same number as a total budget is below the floor cost 42, so
    # every block stops at the lowest candidate.
    assert greedy_allocate(COUNTS, SCORES, budget=7, budget_is_average=True) == [7, 7, 7]
    assert greedy_allocate(COUNTS, SCORES, budget=7, budget_is_average=False) == [3, 3, 3]
