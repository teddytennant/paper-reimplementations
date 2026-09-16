//! Threshold algorithm for random-order triangle counting
//! (Etgar, Gilbert, Liu, McGregor, arXiv:2609.15848).
//!
//! Algorithm 1 (Section 2) reads a random-order edge stream until `Q`
//! triangles are observed in the prefix and returns `T̂ = Q (m/S)^3`.
//! If the stream ends before `Q` triangles, the estimate is the exact
//! prefix count (equal to `T` when the stream is all of `E`).
//!
//! Section 3's independent-edge estimator `T̂ = Y / p^3` is included
//! because the analysis uses it.

pub mod graph;
pub mod sampling;
pub mod threshold;

pub use graph::{eta, exact_triangle_count, tau, Edge, Graph, Vertex};
pub use sampling::independent_edge_estimate;
pub use threshold::{
    random_order_edges, theorem11_q, theorem11_q_with_c0, threshold_estimate, ThresholdResult,
    THEOREM11_C0,
};
