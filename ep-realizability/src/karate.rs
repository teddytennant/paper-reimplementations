//! Zachary's karate club: 34 vertices, 78 undirected edges (0-indexed).

use crate::graph::SimpleGraph;

/// Number of vertices.
pub const KARATE_N: usize = 34;

/// Zachary (1977) edge list, 0-indexed, each pair `u < v`.
pub const KARATE_EDGES: &[(usize, usize)] = &[
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

/// Instructor / "officer club" faction used in the reconstruction test
/// (0-indexed). The complement is the administrator faction.
pub const KARATE_FACTION_A: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21,
];

pub fn karate_club() -> SimpleGraph {
    SimpleGraph::from_edges(KARATE_N, KARATE_EDGES.iter().copied()).expect("karate is simple")
}

/// Cell id 0 on [`KARATE_FACTION_A`], cell id 1 on the complement.
pub fn karate_faction_partition() -> Vec<usize> {
    let mut p = vec![1usize; KARATE_N];
    for &v in KARATE_FACTION_A {
        p[v] = 0;
    }
    p
}

