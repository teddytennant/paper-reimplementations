//! Undirected connected simple graphs as adjacency lists.

use std::fmt;

/// Undirected simple graph with vertices `0..n`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    adj: Vec<Vec<usize>>,
}

impl Graph {
    /// Empty `n`-vertex graph (no edges).
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
        }
    }

    /// Number of vertices.
    pub fn n(&self) -> usize {
        self.adj.len()
    }

    /// Add an undirected edge. Ignores loops and duplicate edges.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u == v {
            return;
        }
        if !self.adj[u].contains(&v) {
            self.adj[u].push(v);
            self.adj[v].push(u);
        }
    }

    /// Neighbors of `v`.
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    /// Path `P_n` on vertices `0-1-...-(n-1)`.
    pub fn path(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1);
        }
        g
    }

    /// Cycle `C_n` on vertices `0-1-...-(n-1)-0`. Requires `n >= 3`.
    pub fn cycle(n: usize) -> Self {
        assert!(n >= 3);
        let mut g = Self::path(n);
        g.add_edge(0, n - 1);
        g
    }

    /// Star `K_{1,q}` with center `0` and leaves `1..=q`. `n = q + 1` vertices.
    pub fn star(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 1..n {
            g.add_edge(0, i);
        }
        g
    }

    /// Complete bipartite `K_{p,q}` with left part `0..p` and right part `p..p+q`.
    pub fn complete_bipartite(p: usize, q: usize) -> Self {
        let mut g = Self::new(p + q);
        for i in 0..p {
            for j in 0..q {
                g.add_edge(i, p + j);
            }
        }
        g
    }

    /// Decode a Prüfer code of length `n-2` over `{0..n}` into a labeled tree.
    pub fn from_prufer(code: &[usize]) -> Self {
        let n = code.len() + 2;
        let mut degree = vec![1usize; n];
        for &c in code {
            degree[c] += 1;
        }
        let mut g = Self::new(n);
        for &s in code {
            let leaf = (0..n).find(|&v| degree[v] == 1).expect("Prüfer leaf");
            g.add_edge(leaf, s);
            degree[leaf] -= 1;
            degree[s] -= 1;
        }
        let rest: Vec<usize> = (0..n).filter(|&v| degree[v] == 1).collect();
        assert_eq!(rest.len(), 2);
        g.add_edge(rest[0], rest[1]);
        g
    }

    /// Two triangles sharing vertex 0 (cactus).
    pub fn bowtie() -> Self {
        let mut g = Self::new(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g.add_edge(0, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 0);
        g
    }

    /// Two triangles joined by a bridge `0-3`.
    pub fn two_cycles_bridge() -> Self {
        let mut g = Self::new(6);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g.add_edge(0, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 5);
        g.add_edge(5, 3);
        g
    }

    /// Complete binary tree of height 2 (7 vertices), root 0.
    pub fn balanced_binary7() -> Self {
        let mut g = Self::new(7);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(2, 5);
        g.add_edge(2, 6);
        g
    }

    /// Root 0 with two children; one child has two leaves (5 vertices).
    pub fn binaryish5() -> Self {
        let mut g = Self::new(5);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g
    }

    /// Undirected edge list, each pair once with `u < v`.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for u in 0..self.n() {
            for &v in &self.adj[u] {
                if u < v {
                    e.push((u, v));
                }
            }
        }
        e
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n={} edges={:?}", self.n(), self.edges())
    }
}
