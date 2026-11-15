//! Temporal bicliques and cliques, and the reduction between them.
//!
//! Follows the conventions of Section 2 of arXiv:2609.02851: every edge carries
//! exactly one time label, so a spanner is just a set of edges and no choice of
//! labels has to be recorded.

/// A time label. Labels are positive integers.
pub type Label = u32;

/// A temporal biclique `G = (S, T, lambda)` with `|S| = n_sources`, `|T| = n_targets`.
///
/// The edge set is implicit, `{{s, t} : s in S, t in T}`, so only the labels are
/// stored, row major with the source as the row.
#[derive(Clone, Debug)]
pub struct TemporalBiclique {
    n_sources: usize,
    n_targets: usize,
    labels: Vec<Label>,
}

impl TemporalBiclique {
    /// Build from a row major label matrix of `n_sources * n_targets` entries.
    pub fn from_labels(n_sources: usize, n_targets: usize, labels: Vec<Label>) -> Self {
        assert_eq!(
            labels.len(),
            n_sources * n_targets,
            "label matrix has the wrong length"
        );
        TemporalBiclique {
            n_sources,
            n_targets,
            labels,
        }
    }

    pub fn n_sources(&self) -> usize {
        self.n_sources
    }

    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// `lambda({s, t})`.
    #[inline]
    pub fn label(&self, s: usize, t: usize) -> Label {
        self.labels[s * self.n_targets + t]
    }

    /// The largest label in the graph, the lifetime `L` of Section 4.
    pub fn lifetime(&self) -> Label {
        self.labels.iter().copied().max().unwrap_or(1)
    }

    /// `true` when `|S| = |T|`, which is what the paper calls balanced.
    pub fn is_balanced(&self) -> bool {
        self.n_sources == self.n_targets
    }

    /// All source indices, in order.
    pub fn all_sources(&self) -> Vec<usize> {
        (0..self.n_sources).collect()
    }

    /// All target indices, in order.
    pub fn all_targets(&self) -> Vec<usize> {
        (0..self.n_targets).collect()
    }
}

/// A temporal clique on `n` vertices with one label per edge.
#[derive(Clone, Debug)]
pub struct TemporalClique {
    n: usize,
    labels: Vec<Label>,
}

impl TemporalClique {
    /// Build from a symmetric row major matrix; the diagonal is ignored.
    pub fn from_labels(n: usize, labels: Vec<Label>) -> Self {
        assert_eq!(labels.len(), n * n, "label matrix has the wrong length");
        for u in 0..n {
            for v in 0..u {
                assert_eq!(
                    labels[u * n + v],
                    labels[v * n + u],
                    "label matrix is not symmetric"
                );
            }
        }
        TemporalClique { n, labels }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `lambda({u, v})` for `u != v`.
    #[inline]
    pub fn label(&self, u: usize, v: usize) -> Label {
        debug_assert_ne!(u, v);
        self.labels[u * self.n + v]
    }

    pub fn lifetime(&self) -> Label {
        let mut m = 1;
        for u in 0..self.n {
            for v in 0..self.n {
                if u != v && self.labels[u * self.n + v] > m {
                    m = self.labels[u * self.n + v];
                }
            }
        }
        m
    }

    /// All `n(n-1)/2` edges, as pairs `u < v`.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.n * self.n.saturating_sub(1) / 2);
        for u in 0..self.n {
            for v in (u + 1)..self.n {
                out.push((u, v));
            }
        }
        out
    }
}

/// Observation 3: the biclique `B` built from a temporal clique `G`.
///
/// `S` and `T` are copies of `V`, and `lambda'({u_S, v_T})` is `lambda({u, v})`
/// when `u != v` and `1` on the diagonal.
pub fn clique_to_biclique(g: &TemporalClique) -> TemporalBiclique {
    let n = g.n();
    let mut labels = vec![1 as Label; n * n];
    for u in 0..n {
        for v in 0..n {
            if u != v {
                labels[u * n + v] = g.label(u, v);
            }
        }
    }
    TemporalBiclique::from_labels(n, n, labels)
}

/// Observation 3, second half: pull a bispanner of `B` back to a spanner of `G`.
///
/// `E_G' = {{u, v} : u != v, {u_S, v_T} in E_B'}`. A 3-hop path in `B` maps to a
/// path in `G` of no more edges, since consecutive copies of the same vertex are
/// merged, so this is a 3-spanner and `|E_G'| <= |E_B'|`.
pub fn lift_bispanner(bispanner: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = bispanner
        .iter()
        .filter(|(u, v)| u != v)
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}
