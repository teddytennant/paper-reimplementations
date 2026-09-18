//! Directed Independent Cascade graph.

/// A directed edge with Independent Cascade activation probability `p` in (0, 1].
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub p: f64,
}

/// Directed graph `G = (V, E)` with IC probabilities `p_e`.
#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    edges: Vec<Edge>,
}

impl Graph {
    /// Create an empty directed graph on `n` vertices labelled `0..n`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    /// Add a directed edge `from -> to` with probability `p`.
    ///
    /// `p` must lie in `(0, 1]`.
    pub fn add_edge(&mut self, from: usize, to: usize, p: f64) {
        assert!(from < self.n && to < self.n, "endpoint out of range");
        assert!(from != to, "self-loops are not used");
        assert!(p > 0.0 && p <= 1.0, "p_e must lie in (0, 1]");
        self.edges.push(Edge { from, to, p });
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Unique undirected pairs `{u, v}` underlying the directed edges.
    pub fn undirected_edges(&self) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = self
            .edges
            .iter()
            .map(|e| {
                if e.from < e.to {
                    (e.from, e.to)
                } else {
                    (e.to, e.from)
                }
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}
