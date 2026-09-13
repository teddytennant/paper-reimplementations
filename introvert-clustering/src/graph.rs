//! Undirected simple graphs and clusterings (Definitions 1 and 2).

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidVertex { vertex: usize, n: usize },
    SelfLoop { vertex: usize },
    DuplicateEdge { u: usize, v: usize },
    InvalidEps { eps: String },
    InvalidBeta { beta: String },
    InvalidLambda { lambda: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidVertex { vertex, n } => {
                write!(f, "vertex {vertex} is outside 0..{n}")
            }
            Error::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex}"),
            Error::DuplicateEdge { u, v } => write!(f, "duplicate edge {u}-{v}"),
            Error::InvalidEps { eps } => {
                write!(f, "eps={eps} is outside (0, 1/2)")
            }
            Error::InvalidBeta { beta } => {
                write!(f, "beta={beta} is outside (0, 1)")
            }
            Error::InvalidLambda { lambda } => {
                write!(f, "lambda={lambda} is outside (0, 1]")
            }
        }
    }
}

impl std::error::Error for Error {}

pub fn validate_eps(eps: f64) -> Result<(), Error> {
    if eps > 0.0 && eps < 0.5 && eps.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidEps {
            eps: format!("{eps}"),
        })
    }
}

pub fn validate_beta(beta: f64) -> Result<(), Error> {
    if beta > 0.0 && beta < 1.0 && beta.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidBeta {
            beta: format!("{beta}"),
        })
    }
}

pub fn validate_lambda(lambda: f64) -> Result<(), Error> {
    if lambda > 0.0 && lambda <= 1.0 && lambda.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidLambda {
            lambda: format!("{lambda}"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    adj: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            edges: Vec::new(),
        }
    }

    /// Build a simple graph on `n` vertices from undirected edges `(u, v)`.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Result<Self, Error> {
        let mut g = Self::new(n);
        for &(u, v) in edges {
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

    pub fn degree(&self, v: usize) -> usize {
        self.adj.get(v).map(|a| a.len()).unwrap_or(0)
    }

    pub fn neighbors(&self, v: usize) -> &[usize] {
        self.adj.get(v).map(|a| a.as_slice()).unwrap_or(&[])
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    pub fn add_edge(&mut self, u: usize, v: usize) -> Result<(), Error> {
        if u >= self.n {
            return Err(Error::InvalidVertex {
                vertex: u,
                n: self.n,
            });
        }
        if v >= self.n {
            return Err(Error::InvalidVertex {
                vertex: v,
                n: self.n,
            });
        }
        if u == v {
            return Err(Error::SelfLoop { vertex: u });
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if self.adj[a].contains(&b) {
            return Err(Error::DuplicateEdge { u: a, v: b });
        }
        self.adj[u].push(v);
        self.adj[v].push(u);
        self.edges.push((a, b));
        Ok(())
    }

    /// Unweighted distances from `src`. `None` means a different component.
    pub fn bfs_dist(&self, src: usize) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.n];
        if src >= self.n {
            return dist;
        }
        let mut q = std::collections::VecDeque::new();
        dist[src] = Some(0);
        q.push_back(src);
        while let Some(u) = q.pop_front() {
            let du = dist[u].expect("queued vertex has a distance");
            for &w in &self.adj[u] {
                if dist[w].is_none() {
                    dist[w] = Some(du + 1);
                    q.push_back(w);
                }
            }
        }
        dist
    }
}

/// Vertex-to-cluster assignment. `None` means unclustered (Definition 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clustering {
    assignment: Vec<Option<usize>>,
}

impl Clustering {
    pub fn new(n: usize) -> Self {
        Self {
            assignment: vec![None; n],
        }
    }

    pub fn from_assignment(assignment: Vec<Option<usize>>) -> Self {
        Self { assignment }
    }

    pub fn n(&self) -> usize {
        self.assignment.len()
    }

    pub fn assignment(&self) -> &[Option<usize>] {
        &self.assignment
    }

    pub fn cluster_of(&self, v: usize) -> Option<usize> {
        self.assignment.get(v).copied().flatten()
    }

    pub(crate) fn uncluster(&mut self, v: usize) {
        if v < self.assignment.len() {
            self.assignment[v] = None;
        }
    }

    /// Nonempty clusters as vertex lists, ordered by cluster id.
    pub fn clusters(&self) -> Vec<Vec<usize>> {
        let mut max_id: Option<usize> = None;
        for id in self.assignment.iter().flatten() {
            max_id = Some(max_id.map_or(*id, |m| m.max(*id)));
        }
        let Some(max_id) = max_id else {
            return Vec::new();
        };
        let mut buckets = vec![Vec::new(); max_id + 1];
        for (v, id) in self.assignment.iter().enumerate() {
            if let Some(c) = id {
                buckets[*c].push(v);
            }
        }
        buckets.into_iter().filter(|c| !c.is_empty()).collect()
    }

    pub fn is_intra_cluster(&self, u: usize, v: usize) -> bool {
        match (self.cluster_of(u), self.cluster_of(v)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    pub fn inter_cluster_edge_count(&self, g: &Graph) -> usize {
        g.edges()
            .iter()
            .filter(|&&(u, v)| !self.is_intra_cluster(u, v))
            .count()
    }

    /// Definition 1 introvert property: |N(v) ∩ C| >= (1/2 - eps) deg(v) for every clustered v.
    pub fn introvert_ok(&self, g: &Graph, eps: f64) -> bool {
        let n = self.n().min(g.n());
        for v in 0..n {
            if self.cluster_of(v).is_none() {
                continue;
            }
            if !vertex_introvert_ok(g, self, v, eps) {
                return false;
            }
        }
        true
    }

    /// Weak diameter in G (paths may leave the cluster). `None` if infinite.
    pub fn weak_diameter(&self, g: &Graph) -> Option<usize> {
        cluster_diameter(self, g, false)
    }

    /// Strong diameter in G[C] (paths stay inside the cluster). `None` if infinite.
    pub fn strong_diameter(&self, g: &Graph) -> Option<usize> {
        cluster_diameter(self, g, true)
    }
}

pub(crate) fn vertex_introvert_ok(g: &Graph, c: &Clustering, v: usize, eps: f64) -> bool {
    let Some(cid) = c.cluster_of(v) else {
        return true;
    };
    let deg = g.degree(v);
    let din = g
        .neighbors(v)
        .iter()
        .filter(|&&w| c.cluster_of(w) == Some(cid))
        .count();
    // Definition 1 / Lemma 1: din >= (1/2 - eps) deg, including isolates (0 >= 0).
    (din as f64) >= (0.5 - eps) * (deg as f64)
}

pub(crate) fn din_in_cluster(g: &Graph, c: &Clustering, v: usize) -> usize {
    let Some(cid) = c.cluster_of(v) else {
        return 0;
    };
    g.neighbors(v)
        .iter()
        .filter(|&&w| c.cluster_of(w) == Some(cid))
        .count()
}

fn cluster_diameter(clustering: &Clustering, g: &Graph, strong: bool) -> Option<usize> {
    let mut max_d = 0usize;
    for cluster in clustering.clusters() {
        if cluster.len() <= 1 {
            continue;
        }
        for &s in &cluster {
            let dist = if strong {
                bfs_in_cluster(g, clustering, s)
            } else {
                g.bfs_dist(s)
            };
            for &t in &cluster {
                match dist.get(t).copied().flatten() {
                    Some(d) => max_d = max_d.max(d),
                    None => return None,
                }
            }
        }
    }
    Some(max_d)
}

fn bfs_in_cluster(g: &Graph, clustering: &Clustering, src: usize) -> Vec<Option<usize>> {
    let n = g.n();
    let mut dist = vec![None; n];
    let Some(cid) = clustering.cluster_of(src) else {
        return dist;
    };
    if src >= n {
        return dist;
    }
    let mut q = std::collections::VecDeque::new();
    dist[src] = Some(0);
    q.push_back(src);
    while let Some(u) = q.pop_front() {
        let du = dist[u].expect("queued vertex has a distance");
        for &w in g.neighbors(u) {
            if clustering.cluster_of(w) != Some(cid) || dist[w].is_some() {
                continue;
            }
            dist[w] = Some(du + 1);
            q.push_back(w);
        }
    }
    dist
}
