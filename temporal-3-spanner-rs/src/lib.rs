//! Temporal 3-spanners of temporal cliques and bicliques, following
//! "Almost Linear 3-Spanners of Temporal Cliques" (arXiv:2609.02851).
//!
//! The core is `Zorro` (Algorithm 1) and the recursion `RecSpan_f`
//! (Algorithm 2) on temporal bicliques, which give a 3-bispanner of size at most
//! `n^(1 + 2/sqrt(ln n))` (Theorem 4). Section 4's variant gives `12 L n` when
//! every label lies in `{1, ..., L}` (Theorem 12). Observation 3 carries both
//! results from bicliques back to cliques.
//!
//! An alpha-spanner approximates the minimum-hop temporal distance within a
//! factor alpha. In a temporal clique every pair is joined by an edge, so a
//! 3-spanner is exactly a subgraph with a temporal path of at most 3 hops
//! between every ordered pair.

pub mod bounded;
pub mod edges;
pub mod graph;
pub mod reach;
pub mod recspan;
pub mod rng;
mod util;
pub mod zorro;

pub use bounded::{bounded_size_bound, bounded_span};
pub use edges::EdgeSet;
pub use graph::{clique_to_biclique, lift_bispanner, Label, TemporalBiclique, TemporalClique};
pub use recspan::{default_f, rec_span, rec_span_with, size_bound};
pub use zorro::{zorro, ZorroOut};

/// A temporal 3-spanner of a temporal clique, via Observation 3 and Theorem 4.
///
/// Returns unordered edges `(u, v)` with `u < v`. Size at most
/// `n^(1 + 2/sqrt(ln n))`.
pub fn clique_spanner(g: &TemporalClique) -> Vec<(usize, usize)> {
    let b = clique_to_biclique(g);
    lift_bispanner(&rec_span(&b).to_vec())
}

/// A temporal 3-spanner of a temporal clique of lifetime at most `lifetime`, via
/// Observation 3 and Theorem 12. Size at most `12 * lifetime * n`.
pub fn clique_spanner_bounded(g: &TemporalClique, lifetime: Label) -> Vec<(usize, usize)> {
    let b = clique_to_biclique(g);
    lift_bispanner(&bounded_span(&b, lifetime).to_vec())
}

/// A uniformly random temporal clique with labels in `1..=max_label`.
pub fn random_clique(n: usize, max_label: Label, seed: u64) -> TemporalClique {
    let mut r = rng::Rng::new(seed);
    let mut labels = vec![1 as Label; n * n];
    for u in 0..n {
        for v in (u + 1)..n {
            let lab = 1 + r.below(max_label as u64) as Label;
            labels[u * n + v] = lab;
            labels[v * n + u] = lab;
        }
    }
    TemporalClique::from_labels(n, labels)
}

/// A uniformly random balanced temporal biclique with labels in `1..=max_label`.
pub fn random_biclique(n: usize, max_label: Label, seed: u64) -> TemporalBiclique {
    let mut r = rng::Rng::new(seed);
    let labels: Vec<Label> = (0..n * n)
        .map(|_| 1 + r.below(max_label as u64) as Label)
        .collect();
    TemporalBiclique::from_labels(n, n, labels)
}
