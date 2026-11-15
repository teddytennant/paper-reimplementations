//! Equitable Partition realizability for dynamics-preserving and privacy-aware
//! network reconstruction (Porcedda, arXiv:2609.16762).
//!
//! Algorithm 1 constructs a simple graph from an exact EP `(AH, H)` by
//! heap-based Havel–Hakimi (diagonal layers) and heap-based bipartite
//! Havel–Hakimi (off-diagonal layers), after Erdős–Gállai / Gale–Ryser tests.
//! Algorithm 2 (`Φ_ε`) selects an integer profile from a noisy quotient.

mod ep;
mod graph;
mod graphic;
mod karate;

pub use ep::{
    ah_from_q, edge_overlap, ep_of, implied_edges, is_equitable, layer_preserving_switch,
    partition_from_h, phi_eps, predicted_edge_overlap, q_from_ah, randomize_ep, realizable,
    realize, realize_sampled, AhMatrix, CellSizes, QMatrix,
};
pub use graph::{Error, SimpleGraph};
pub use graphic::{
    bipartite_havel_hakimi, erdos_gallai, gale_ryser, havel_hakimi, is_graphic_regular,
};
pub use karate::{
    karate_club, karate_faction_partition, KARATE_EDGES, KARATE_FACTION_A, KARATE_N,
};
