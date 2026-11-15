"""Tests for the signed fractional hypertree width enumerator.

The oracle below is a separate brute-force evaluator. It does not call
solve_query, BuildEnumDS, or any skip-link routine.
"""

import itertools

from sfhw import fhw, make_query, rho_star, sfhw, solve_query


def brute_force(variables, positive, negative):
    """Enumerate the domain product and keep tuples that match every positive
    atom and no negative atom.

    Domains are the values that appear for each variable in the given atoms.
    """
    pos = [(tuple(vars_), {tuple(tup) for tup in tuples}) for _, vars_, tuples in positive]
    neg = [(tuple(vars_), {tuple(tup) for tup in tuples}) for _, vars_, tuples in negative]
    domains = {var: set() for var in variables}
    for vars_, tuples in pos + neg:
        for tup in tuples:
            for var, val in zip(vars_, tup):
                domains[var].add(val)
    answers = []
    domain_lists = [sorted(domains[var], key=repr) for var in variables]
    for combo in itertools.product(*domain_lists):
        assign = dict(zip(variables, combo))
        if any(tuple(assign[var] for var in vars_) not in tuples for vars_, tuples in pos):
            continue
        if any(tuple(assign[var] for var in vars_) in tuples for vars_, tuples in neg):
            continue
        answers.append({var: assign[var] for var in variables})
    return answers


def _as_set(rows, variables):
    return {tuple(row[var] for var in variables) for row in rows}


def _assert_matches(positive, negative, order, variables=None):
    query = make_query(positive, negative)
    variables = list(variables or query.variables)
    got = solve_query(query, order)
    expected = brute_force(variables, positive, negative)
    got_set = _as_set(got, variables)
    exp_set = _as_set(expected, variables)
    assert got_set == exp_set
    assert len(got) == len(got_set)


def test_positive_only_join_matches_brute_force_once():
    positive = [
        ("R", ("x", "y"), [(1, 1), (1, 2), (2, 2), (3, 1)]),
        ("S", ("y", "z"), [(1, 4), (1, 5), (2, 4)]),
    ]
    _assert_matches(positive, [], ["x", "y", "z"])
    # A positive triangle, so the bag join is not a path.
    positive_tri = [
        ("R", ("x", "y"), [(1, 1), (1, 2), (2, 2)]),
        ("S", ("y", "z"), [(1, 1), (2, 2), (2, 3)]),
        ("T", ("x", "z"), [(1, 1), (2, 2), (1, 3)]),
    ]
    _assert_matches(positive_tri, [], ["x", "y", "z"])


def test_paper_shape_with_a_skip():
    # Q(x,y,z,w) <- R(x,y), not T(x,y,z), not W(x,w), U(z), V(w)
    # Instance shaped like Example 4.3. In the b2 list the order is
    # a1, a2, a4, a5, a6. W blocks a2 (label d1) while a1 stays; T blocks
    # a4 and a5 (label c1) while a6 stays. Those are the skips in the figure.
    positive = [
        (
            "R",
            ("x", "y"),
            [
                ("a1", "b1"),
                ("a2", "b1"),
                ("a3", "b1"),
                ("a1", "b2"),
                ("a2", "b2"),
                ("a4", "b2"),
                ("a5", "b2"),
                ("a6", "b2"),
            ],
        ),
        ("U", ("z",), [("c1",), ("c2",)]),
        ("V", ("w",), [("d1",)]),
    ]
    negative = [
        ("T", ("x", "y", "z"), [("a4", "b2", "c1"), ("a5", "b2", "c1")]),
        ("W", ("x", "w"), [("a2", "d1")]),
    ]
    variables = ["x", "y", "z", "w"]
    _assert_matches(positive, negative, ["x", "y", "z", "w"], variables)
    answers = _as_set(solve_query(make_query(positive, negative), ["x", "y", "z", "w"]), variables)
    assert ("a2", "b2", "c2", "d1") not in answers
    assert ("a1", "b2", "c2", "d1") in answers
    assert ("a4", "b2", "c1", "d1") not in answers
    assert ("a6", "b2", "c1", "d1") in answers
    # Section 4.4's complete example, every elimination order.
    tri_pos = [
        ("R", ("x",), [("a1",), ("a2",), ("a3",)]),
        ("T", ("y", "z"), [("b1", "c1"), ("b2", "c1"), ("b3", "c2")]),
    ]
    tri_neg = [
        ("A", ("x", "y"), [("a1", "b1"), ("a1", "b2"), ("a2", "b2"), ("a3", "b2")]),
        ("B", ("x", "z"), [("a2", "c1"), ("a3", "c1"), ("a2", "c2")]),
    ]
    for order in itertools.permutations(["x", "y", "z"]):
        _assert_matches(tri_pos, tri_neg, order, ["x", "y", "z"])


def test_blocked_prefix_is_not_emitted():
    # y=1 is a positive value, but every x in R is blocked by T for that y.
    # The prefix y=1 must not appear in any emitted tuple. y=2 still extends.
    positive = [
        ("R", ("x",), [(1,), (2,), (3,)]),
        ("S", ("y",), [(1,), (2,)]),
    ]
    negative = [
        ("T", ("x", "y"), [(1, 1), (2, 1), (3, 1)]),
    ]
    _assert_matches(positive, negative, ["y", "x"])
    answers = solve_query(make_query(positive, negative), ["y", "x"])
    assert answers
    assert all(row["y"] != 1 for row in answers)
    assert any(row["y"] == 2 for row in answers)


def test_answers_are_unique():
    positive = [
        ("R", ("x", "y"), [(1, 1), (1, 2), (2, 1), (2, 2)]),
        ("S", ("y", "z"), [(1, 1), (2, 1), (2, 2)]),
    ]
    negative = [
        ("T", ("x", "z"), [(1, 1), (2, 2)]),
        ("W", ("x", "y"), [(2, 1)]),
    ]
    got = solve_query(make_query(positive, negative), ["x", "y", "z"])
    keys = [tuple(sorted(row.items())) for row in got]
    assert len(keys) == len(set(keys))
    assert len(got) == len(brute_force(["x", "y", "z"], positive, negative))


def test_sfhw_of_positive_tree_is_one_and_equals_fhw():
    # Path R(x,y), S(y,z). Order (x, y, z):
    #   chi(x) = {x,y}, covered by R with weight 1, rho* = 1
    #   chi(y) = {y,z}, covered by S with weight 1, rho* = 1
    #   chi(z) = {z}, covered by S (it contains z), rho* = 1
    # Width of this order is 1, so fhw = sfhw = 1.
    # Order (y, x, z) is worse (chi(y) = {x,y,z}, rho* = 2) but the min is 1.
    query = make_query(
        [
            ("R", ("x", "y"), [(1, 1), (1, 2)]),
            ("S", ("y", "z"), [(1, 3), (2, 4)]),
        ],
        [],
    )
    assert sfhw(query) == 1
    assert fhw(query) == 1
    assert sfhw(query) == fhw(query)


def test_sfhw_of_positive_edge_is_one():
    # Single edge R(x,y). chi(x) = {x,y} has rho* = 1.
    # chi(y) = {y} is covered by R, which contains y, so rho* = 1.
    query = make_query([("R", ("x", "y"), [(1, 2), (3, 4)])], [])
    assert sfhw(query) == 1
    assert fhw(query) == 1
    assert rho_star([{"x", "y"}], {"x", "y"}) == 1
    assert rho_star([{"x", "y"}, {"y", "z"}, {"x", "z"}], {"x", "y", "z"}) == 1.5
    # Two edges {x,y} and {y,z} need weight 1 each to cover x and z, so rho* = 2.
    assert rho_star([{"x", "y"}, {"y", "z"}], {"x", "y", "z"}) == 2


def test_signed_acyclic_has_sfhw_one_and_a_non_example_is_larger():
    # Signed-acyclic: the negative edge is the same variable set as the
    # positive edge. For S empty and S = {S}, H^S has edge {x,y}
    # (the negative edge duplicates R). Same bags as the single-edge query:
    # chi(x) = {x,y} rho* = 1, chi(y) = {y} rho* = 1. So sfhw = 1.
    acyclic = make_query(
        [("R", ("x", "y"), [(1, 1), (1, 2), (2, 2)])],
        [("S", ("x", "y"), [(1, 1)])],
    )
    assert sfhw(acyclic) == 1

    # Not signed-acyclic (Example 3.8): R(x,y), T(y,z), not S(x,z).
    # For S = {S}, H^S is a triangle. Whichever variable is first, its bag
    # is the union of the two edges that contain it, hence all three vertices.
    # rho* of the triangle is 3/2: weight 1/2 on each edge covers every vertex
    # to 1, and any two edges need total weight 2 (each private vertex forces
    # weight 1 on its only edge). Later bags have size at most 2, rho* = 1.
    # Every order therefore has width 3/2, so sfhw = 3/2 > 1.
    cyclic = make_query(
        [
            ("R", ("x", "y"), [(1, 1), (1, 2)]),
            ("T", ("y", "z"), [(1, 1), (2, 2)]),
        ],
        [("S", ("x", "z"), [(1, 2)])],
    )
    assert sfhw(cyclic) == 1.5
    assert sfhw(cyclic) > 1
