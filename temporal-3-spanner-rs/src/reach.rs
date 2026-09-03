//! Minimum-hop temporal distances, computed from nothing but an edge list.
//!
//! This is the checker the tests use to decide whether a returned edge set is a
//! 3-spanner. It knows nothing about `Zorro` or the recursion; it takes labelled
//! edges and does its own search.

use crate::graph::Label;

const INF: Label = Label::MAX;

/// An undirected temporal graph as an adjacency list of labelled edges.
#[derive(Clone, Debug)]
pub struct TemporalGraph {
    n: usize,
    adj: Vec<Vec<(usize, Label)>>,
}

impl TemporalGraph {
    pub fn new(n: usize) -> Self {
        TemporalGraph {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, label: Label) {
        assert!(label < INF);
        self.adj[u].push((v, label));
        self.adj[v].push((u, label));
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Minimum number of hops of a temporal path from `u` to each vertex, or
    /// `None` past `max_hops`.
    ///
    /// Round `h` keeps, for every vertex, the earliest arrival time reachable in
    /// at most `h` hops. Earliest arrival dominates: anything you can do after
    /// arriving at time `a'` you can also do after arriving at `a <= a'`. A
    /// minimum-hop temporal walk cannot repeat a vertex, since cutting the loop
    /// out of a nondecreasing label sequence leaves it nondecreasing, so this
    /// also gives the minimum-hop temporal path.
    pub fn hop_distances_from(&self, u: usize, max_hops: usize) -> Vec<Option<usize>> {
        let mut arrive = vec![INF; self.n];
        arrive[u] = 0;
        let mut dist = vec![None; self.n];
        dist[u] = Some(0);
        for h in 1..=max_hops {
            let mut next = arrive.clone();
            for (w, out) in self.adj.iter().enumerate() {
                if arrive[w] == INF {
                    continue;
                }
                for &(v, lab) in out {
                    if lab >= arrive[w] && lab < next[v] {
                        next[v] = lab;
                    }
                }
            }
            for v in 0..self.n {
                if dist[v].is_none() && next[v] != INF {
                    dist[v] = Some(h);
                }
            }
            if next == arrive {
                break;
            }
            arrive = next;
        }
        dist
    }
}

/// Build a `TemporalGraph` from clique edges given as unordered pairs.
pub fn clique_graph(
    n: usize,
    edges: &[(usize, usize)],
    label: impl Fn(usize, usize) -> Label,
) -> TemporalGraph {
    let mut g = TemporalGraph::new(n);
    for &(u, v) in edges {
        g.add_edge(u, v, label(u, v));
    }
    g
}

/// Build a `TemporalGraph` from biclique edges. Sources keep their index,
/// target `t` becomes vertex `n_sources + t`.
pub fn biclique_graph(
    n_sources: usize,
    n_targets: usize,
    edges: &[(usize, usize)],
    label: impl Fn(usize, usize) -> Label,
) -> TemporalGraph {
    let mut g = TemporalGraph::new(n_sources + n_targets);
    for &(s, t) in edges {
        g.add_edge(s, n_sources + t, label(s, t));
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nondecreasing_labels_make_a_path() {
        let mut g = TemporalGraph::new(4);
        g.add_edge(0, 1, 2);
        g.add_edge(1, 2, 5);
        g.add_edge(2, 3, 5);
        let d = g.hop_distances_from(0, 4);
        assert_eq!(d, vec![Some(0), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn decreasing_labels_do_not() {
        let mut g = TemporalGraph::new(3);
        g.add_edge(0, 1, 9);
        g.add_edge(1, 2, 3);
        let d = g.hop_distances_from(0, 3);
        assert_eq!(d, vec![Some(0), Some(1), None]);
        // Backwards the same edges are nondecreasing.
        assert_eq!(g.hop_distances_from(2, 3), vec![Some(2), Some(1), Some(0)]);
    }

    #[test]
    fn the_earliest_arrival_is_the_one_that_matters() {
        // Reaching 1 at time 9 is a dead end; reaching it at time 1 through the
        // longer route is not, and the search has to notice.
        let mut g = TemporalGraph::new(4);
        g.add_edge(0, 1, 9);
        g.add_edge(0, 2, 1);
        g.add_edge(2, 1, 1);
        g.add_edge(1, 3, 4);
        let d = g.hop_distances_from(0, 4);
        assert_eq!(d[1], Some(1));
        assert_eq!(d[3], Some(3));
    }

    #[test]
    fn a_direct_edge_is_one_hop() {
        let mut g = TemporalGraph::new(2);
        g.add_edge(0, 1, 7);
        assert_eq!(g.hop_distances_from(0, 2), vec![Some(0), Some(1)]);
    }
}
