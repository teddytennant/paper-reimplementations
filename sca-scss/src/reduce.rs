//! Terminal graph J (Theorem 1.1) and the unweighted kernel of Theorem 1.4.

use std::collections::BTreeSet;

use crate::distance::{layered_at_most, min_link_path, unweighted_shortest, INF};
use crate::graph::DiGraph;

/// Unweighted terminal graph: one edge uv of cost d_H(u, v) for every
/// reachable pair of distinct terminals.
#[derive(Clone, Debug)]
pub struct TerminalGraph {
    pub terminals: Vec<usize>,
    pub n: usize,
    pub edges: Vec<(usize, usize, u64)>,
}

/// Two-cost terminal graph: an edge uv of costs (h, d_h(u, v)) whenever finite.
#[derive(Clone, Debug)]
pub struct WeightedTerminalGraph {
    pub terminals: Vec<usize>,
    pub n: usize,
    /// (u, v, ell, weight) with u, v in 0..n
    pub edges: Vec<(usize, usize, u64, u64)>,
}

pub fn build_unweighted(g: &DiGraph, links: &[(usize, usize)]) -> TerminalGraph {
    let terminals = g.terminals();
    let n = terminals.len();
    let mut edges = Vec::new();
    for (i, &u) in terminals.iter().enumerate() {
        let (dist, _) = unweighted_shortest(g, links, u);
        for (j, &v) in terminals.iter().enumerate() {
            if i == j {
                continue;
            }
            if dist[v] < INF {
                edges.push((i, j, dist[v]));
            }
        }
    }
    TerminalGraph {
        terminals,
        n,
        edges,
    }
}

pub fn build_weighted(
    g: &DiGraph,
    links: &[(usize, usize, u64)],
    k: usize,
) -> WeightedTerminalGraph {
    let terminals = g.terminals();
    let n = terminals.len();
    let mut edges = Vec::new();
    for (i, &u) in terminals.iter().enumerate() {
        let table = layered_at_most(g, links, u, k);
        for (j, &v) in terminals.iter().enumerate() {
            if i == j {
                continue;
            }
            for h in 0..=k {
                let w = table[h][v];
                if w < INF {
                    edges.push((i, j, h as u64, w));
                }
            }
        }
    }
    WeightedTerminalGraph {
        terminals,
        n,
        edges,
    }
}

/// Unweighted kernel of Theorem 1.4 (no Frank-Tardos).
///
/// Retains one min-link terminal path per ordered pair, takes W = Z union
/// endpoints of those links, and puts a base edge xy in G_W iff G has an
/// x-to-y path. Returns None on the |S|>k or |T|>k reject.
pub fn kernelize_unweighted(
    g: &DiGraph,
    links: &[(usize, usize)],
    k: usize,
) -> Option<(DiGraph, Vec<(usize, usize)>)> {
    if g.is_strongly_connected() {
        return Some((DiGraph::new(g.len().min(1)), Vec::new()));
    }
    let s = g.sources();
    let t = g.sinks();
    if s.len() > k || t.len() > k {
        return None;
    }
    let z = g.terminals();
    let mut retained: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut wset: BTreeSet<usize> = z.iter().copied().collect();
    for &u in &z {
        for &v in &z {
            if u == v {
                continue;
            }
            if let Some(hops) = min_link_path(g, links, u, v) {
                for (x, y, via_link) in hops {
                    if via_link {
                        retained.insert((x, y));
                        wset.insert(x);
                        wset.insert(y);
                    }
                }
            }
        }
    }
    let wverts: Vec<usize> = wset.into_iter().collect();
    let idx = |x: usize| wverts.binary_search(&x).unwrap();
    let mut gw = DiGraph::new(wverts.len());
    for (i, &x) in wverts.iter().enumerate() {
        let reach = g.reachable_from(x);
        for (j, &y) in wverts.iter().enumerate() {
            if i != j && reach[y] {
                gw.add_edge(i, j);
            }
        }
    }
    let k_links: Vec<(usize, usize)> = retained
        .into_iter()
        .map(|(x, y)| (idx(x), idx(y)))
        .collect();
    Some((gw, k_links))
}
