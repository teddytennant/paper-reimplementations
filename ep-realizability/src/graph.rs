//! Simple undirected loopless graphs.

use std::fmt;

/// Recoverable failure of a graphicality test or a construction step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NotGraphic,
    NotBipartiteGraphic,
    Infeasible,
    DimensionMismatch,
    NotDivisible,
    InvalidDegree,
    MultiEdge,
    SelfLoop,
    OutOfRange,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotGraphic => write!(f, "degree sequence is not graphic"),
            Error::NotBipartiteGraphic => write!(f, "bipartite degree sequence is not graphic"),
            Error::Infeasible => write!(f, "equitable partition profile is not realizable"),
            Error::DimensionMismatch => write!(f, "matrix and cell-size dimensions do not match"),
            Error::NotDivisible => write!(f, "quotient and cell sizes do not divide evenly"),
            Error::InvalidDegree => write!(f, "degree out of range for a simple graph"),
            Error::MultiEdge => write!(f, "duplicate edge"),
            Error::SelfLoop => write!(f, "self-loop"),
            Error::OutOfRange => write!(f, "vertex or cell index out of range"),
        }
    }
}

impl std::error::Error for Error {}

/// Simple undirected loopless graph: no self-loops, no parallel edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleGraph {
    n: usize,
    /// Undirected edges with `u < v`, sorted lexicographically.
    edges: Vec<(usize, usize)>,
    adj: Vec<Vec<usize>>,
}

impl SimpleGraph {
    /// Empty graph on `n` vertices.
    pub fn empty(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            adj: vec![Vec::new(); n],
        }
    }

    /// Build from an undirected edge list, rejecting loops and multi-edges.
    pub fn from_edges<I>(n: usize, edges: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let mut g = Self::empty(n);
        for (u, v) in edges {
            g.add_edge(u, v)?;
        }
        Ok(g)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    pub fn adj(&self) -> &[Vec<usize>] {
        &self.adj
    }

    pub fn degree(&self, u: usize) -> usize {
        self.adj[u].len()
    }

    pub fn degrees(&self) -> Vec<usize> {
        self.adj.iter().map(|nbrs| nbrs.len()).collect()
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u == v || u >= self.n || v >= self.n {
            return false;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.edges.binary_search(&(a, b)).is_ok()
    }

    /// Insert `{u,v}` if missing. Errors on loops, out-of-range indices, or a duplicate.
    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<(), Error> {
        if u == v {
            return Err(Error::SelfLoop);
        }
        if u >= self.n || v >= self.n {
            return Err(Error::OutOfRange);
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        match self.edges.binary_search(&(a, b)) {
            Ok(_) => Err(Error::MultiEdge),
            Err(idx) => {
                self.edges.insert(idx, (a, b));
                insert_sorted(&mut self.adj[a], b);
                insert_sorted(&mut self.adj[b], a);
                Ok(())
            }
        }
    }

    pub fn remove_edge(&mut self, u: usize, v: usize) -> bool {
        if u == v || u >= self.n || v >= self.n {
            return false;
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if let Ok(idx) = self.edges.binary_search(&(a, b)) {
            self.edges.remove(idx);
            remove_sorted(&mut self.adj[a], b);
            remove_sorted(&mut self.adj[b], a);
            true
        } else {
            false
        }
    }

    /// No self-loops, no duplicate undirected edges, all endpoints in range.
    pub fn is_simple(&self) -> bool {
        let mut seen = std::collections::BTreeSet::new();
        for &(u, v) in &self.edges {
            if u >= self.n || v >= self.n || u >= v {
                return false;
            }
            if !seen.insert((u, v)) {
                return false;
            }
        }
        true
    }
}

fn insert_sorted(vec: &mut Vec<usize>, x: usize) {
    match vec.binary_search(&x) {
        Ok(_) => {}
        Err(i) => vec.insert(i, x),
    }
}

fn remove_sorted(vec: &mut Vec<usize>, x: usize) {
    if let Ok(i) = vec.binary_search(&x) {
        vec.remove(i);
    }
}
