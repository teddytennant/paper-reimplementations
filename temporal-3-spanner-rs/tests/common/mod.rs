#![allow(dead_code)]

//! Independent verification helpers shared by the test files.
//!
//! Nothing here calls the construction. Everything goes through
//! `reach::TemporalGraph`, which only ever sees a list of labelled edges.

use temporal_spanner::reach::{biclique_graph, clique_graph, TemporalGraph};
use temporal_spanner::{TemporalBiclique, TemporalClique};

/// Minimum-hop temporal distance from every vertex, as a matrix of `Option`.
pub fn all_pairs(g: &TemporalGraph) -> Vec<Vec<Option<usize>>> {
    let cap = g.n();
    (0..g.n()).map(|u| g.hop_distances_from(u, cap)).collect()
}

/// Check that `edges` is a temporal 3-bispanner of the balanced biclique `g`:
/// every source reaches every target in at most 3 hops, and at most
/// `3 * d_input`, where `d_input` is 1 for every pair of a biclique.
pub fn check_bispanner(g: &TemporalBiclique, edges: &[(usize, usize)]) {
    let ns = g.n_sources();
    let nt = g.n_targets();
    let full: Vec<(usize, usize)> = (0..ns).flat_map(|s| (0..nt).map(move |t| (s, t))).collect();
    let input = biclique_graph(ns, nt, &full, |s, t| g.label(s, t));
    let span = biclique_graph(ns, nt, edges, |s, t| g.label(s, t));
    let d_in = all_pairs(&input);
    let d_sp = all_pairs(&span);
    for s in 0..ns {
        for t in 0..nt {
            let din = d_in[s][ns + t].expect("input biclique is not temporally connected");
            assert_eq!(
                din, 1,
                "a biclique pair should be one hop apart in the input"
            );
            let dsp = d_sp[s][ns + t]
                .unwrap_or_else(|| panic!("no temporal path from source {s} to target {t}"));
            assert!(dsp <= 3, "source {s} to target {t} takes {dsp} hops");
            assert!(
                dsp <= 3 * din,
                "source {s} to target {t}: {dsp} > 3 * {din}"
            );
        }
    }
}

/// Same for a temporal clique, over every ordered pair of distinct vertices.
pub fn check_spanner(g: &TemporalClique, edges: &[(usize, usize)]) {
    let n = g.n();
    let input = clique_graph(n, &g.edges(), |u, v| g.label(u, v));
    let span = clique_graph(n, edges, |u, v| g.label(u, v));
    let d_in = all_pairs(&input);
    let d_sp = all_pairs(&span);
    for u in 0..n {
        for v in 0..n {
            if u == v {
                continue;
            }
            let din = d_in[u][v].expect("input clique is not temporally connected");
            assert_eq!(din, 1, "a clique pair should be one hop apart in the input");
            let dsp = d_sp[u][v].unwrap_or_else(|| panic!("no temporal path from {u} to {v}"));
            assert!(dsp <= 3, "{u} to {v} takes {dsp} hops");
            assert!(dsp <= 3 * din, "{u} to {v}: {dsp} > 3 * {din}");
        }
    }
}
