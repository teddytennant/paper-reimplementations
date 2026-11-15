//! Undirected edge-weighted graph for Polyamorous Scheduling.
//!
//! Section 1: instance G = (P, E) with positive growth rate g_e on each edge.
//! Vertex growth G_v = ∑_{e ∋ v} g_e, and G* = max_v G_v.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub g: f64,
}

#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    edges: Vec<Edge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidVertex { vertex: usize, n: usize },
    SelfLoop { vertex: usize },
    NonPositiveWeight { weight: String },
    DuplicateEdge { u: usize, v: usize },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::InvalidVertex { vertex, n } => {
                write!(f, "vertex {vertex} is outside 0..{n}")
            }
            GraphError::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex}"),
            GraphError::NonPositiveWeight { weight } => {
                write!(f, "edge weight {weight} is not positive")
            }
            GraphError::DuplicateEdge { u, v } => write!(f, "duplicate edge {u}-{v}"),
        }
    }
}

impl std::error::Error for GraphError {}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn add_edge(&mut self, u: usize, v: usize, g: f64) -> Result<usize, GraphError> {
        if u >= self.n {
            return Err(GraphError::InvalidVertex {
                vertex: u,
                n: self.n,
            });
        }
        if v >= self.n {
            return Err(GraphError::InvalidVertex {
                vertex: v,
                n: self.n,
            });
        }
        if u == v {
            return Err(GraphError::SelfLoop { vertex: u });
        }
        if !(g > 0.0) || !g.is_finite() {
            return Err(GraphError::NonPositiveWeight {
                weight: format!("{g}"),
            });
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if self
            .edges
            .iter()
            .any(|e| (e.u.min(e.v), e.u.max(e.v)) == (a, b))
        {
            return Err(GraphError::DuplicateEdge { u: a, v: b });
        }
        let idx = self.edges.len();
        self.edges.push(Edge { u, v, g });
        Ok(idx)
    }

    /// Section 1: G_v = ∑_{e ∋ v} g_e.
    pub fn vertex_growth(&self) -> Vec<f64> {
        let mut gv = vec![0.0; self.n];
        for e in &self.edges {
            gv[e.u] += e.g;
            gv[e.v] += e.g;
        }
        gv
    }

    /// Section 1: G* = max_v G_v. Empty graphs have G* = 0.
    pub fn g_star(&self) -> f64 {
        self.vertex_growth().into_iter().fold(0.0, f64::max)
    }
}
