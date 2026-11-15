//! Independent-edge-sampling estimator (paper Section 3).

use rand::Rng;

use crate::graph::{exact_triangle_count, Graph};

/// Sample each edge independently with probability `p`, let `Y` be the
/// number of triangles in the sampled graph, return `T̂ = Y / p^3`.
///
/// Unbiased for `T` when `0 < p ≤ 1`. If `p = 0` the estimator is `0`
/// (no edges are kept; the `p^3` denominator is not used).
pub fn independent_edge_estimate<R: Rng + ?Sized>(graph: &Graph, p: f64, rng: &mut R) -> f64 {
    assert!(
        (0.0..=1.0).contains(&p),
        "sampling probability p must be in [0, 1]"
    );
    if p == 0.0 {
        return 0.0;
    }
    let sampled = Graph::from_edges(graph.edges().iter().copied().filter(|_| rng.gen_bool(p)));
    let y = exact_triangle_count(&sampled) as f64;
    y / p.powi(3)
}
