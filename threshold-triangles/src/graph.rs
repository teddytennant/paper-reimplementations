//! Undirected simple graph: edges `(u, v)` with `u < v`, adjacency as hash maps.

use std::collections::{HashMap, HashSet};

/// Vertex identifier.
pub type Vertex = u32;

/// Undirected edge with `u < v`.
pub type Edge = (Vertex, Vertex);

/// Undirected simple graph.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    adj: HashMap<Vertex, HashSet<Vertex>>,
    edges: Vec<Edge>,
    edge_set: HashSet<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from an edge list. Loops are dropped; duplicate edges are kept once.
    pub fn from_edges<I>(edges: I) -> Self
    where
        I: IntoIterator<Item = (Vertex, Vertex)>,
    {
        let mut g = Graph::new();
        for (u, v) in edges {
            g.insert(u, v);
        }
        g
    }

    /// Insert `{u, v}`. Returns true if the edge was new. Ignores loops.
    pub fn insert(&mut self, u: Vertex, v: Vertex) -> bool {
        if u == v {
            return false;
        }
        let e = canon(u, v);
        if !self.edge_set.insert(e) {
            return false;
        }
        self.edges.push(e);
        self.adj.entry(e.0).or_default().insert(e.1);
        self.adj.entry(e.1).or_default().insert(e.0);
        true
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn contains_edge(&self, u: Vertex, v: Vertex) -> bool {
        if u == v {
            return false;
        }
        self.edge_set.contains(&canon(u, v))
    }

    pub fn neighbors(&self, v: Vertex) -> Option<&HashSet<Vertex>> {
        self.adj.get(&v)
    }

    /// `|N(u) ∩ N(v)|`. Equals `τ(e)` when `e = {u, v}` is an edge.
    pub fn common_neighbor_count(&self, u: Vertex, v: Vertex) -> u64 {
        common_count(&self.adj, u, v)
    }

    /// Exact triangle count `T`. Each triangle is counted once.
    pub fn triangle_count(&self) -> u64 {
        let sum: u64 = self
            .edges
            .iter()
            .map(|&(u, v)| self.common_neighbor_count(u, v))
            .sum();
        debug_assert_eq!(sum % 3, 0);
        sum / 3
    }

    /// `τ(e)`: number of triangles containing `e = {u, v}`. Zero if not an edge.
    pub fn tau(&self, u: Vertex, v: Vertex) -> u64 {
        if !self.contains_edge(u, v) {
            return 0;
        }
        self.common_neighbor_count(u, v)
    }

    /// `η = max_e τ(e)`. Zero on an empty graph.
    pub fn eta(&self) -> u64 {
        self.edges
            .iter()
            .map(|&(u, v)| self.common_neighbor_count(u, v))
            .max()
            .unwrap_or(0)
    }
}

/// Canonical undirected edge `(min, max)`.
pub fn canon(u: Vertex, v: Vertex) -> Edge {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

pub(crate) fn common_count(adj: &HashMap<Vertex, HashSet<Vertex>>, u: Vertex, v: Vertex) -> u64 {
    let Some(nu) = adj.get(&u) else {
        return 0;
    };
    let Some(nv) = adj.get(&v) else {
        return 0;
    };
    let (small, large) = if nu.len() <= nv.len() {
        (nu, nv)
    } else {
        (nv, nu)
    };
    small.iter().filter(|w| large.contains(w)).count() as u64
}

/// Exact triangle count `T`.
pub fn exact_triangle_count(graph: &Graph) -> u64 {
    graph.triangle_count()
}

/// `η = max_e τ(e)`.
pub fn eta(graph: &Graph) -> u64 {
    graph.eta()
}

/// `τ({u, v})`.
pub fn tau(graph: &Graph, u: Vertex, v: Vertex) -> u64 {
    graph.tau(u, v)
}
