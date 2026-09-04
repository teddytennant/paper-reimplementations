//! Simple undirected graphs plus the generators the tests run over.

use crate::rng::SplitMix64;

/// An undirected simple graph on the vertex set `0..n`.
///
/// Adjacency lists are sorted, which gives the network layer a cheap
/// `is_neighbor` test and makes every "choose any" tie-break in DCluster fall
/// on the smallest identifier.
#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    adj: Vec<Vec<u32>>,
    m: usize,
}

impl Graph {
    /// Build from an edge list. Self loops and repeated edges are dropped.
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Graph {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            assert!((u as usize) < n && (v as usize) < n, "edge endpoint out of range");
            if u == v {
                continue;
            }
            adj[u as usize].push(v);
            adj[v as usize].push(u);
        }
        let mut m = 0;
        for list in adj.iter_mut() {
            list.sort_unstable();
            list.dedup();
            m += list.len();
        }
        Graph { n, adj, m: m / 2 }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    pub fn neighbors(&self, v: usize) -> &[u32] {
        &self.adj[v]
    }

    pub fn is_neighbor(&self, u: usize, v: usize) -> bool {
        self.adj[u].binary_search(&(v as u32)).is_ok()
    }

    /// Every edge once, with the smaller endpoint first.
    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.m);
        for u in 0..self.n {
            for &v in &self.adj[u] {
                if (u as u32) < v {
                    out.push((u as u32, v));
                }
            }
        }
        out
    }
}

/// Erdos-Renyi `G(n, p)`.
pub fn erdos_renyi(n: usize, p: f64, seed: u64) -> Graph {
    let mut rng = SplitMix64::seed(seed);
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.bernoulli(p) {
                edges.push((u as u32, v as u32));
            }
        }
    }
    Graph::from_edges(n, &edges)
}

/// Barabasi-Albert preferential attachment, `m0` initial clique, `k` edges per
/// arrival. The degree sequence comes out heavy tailed, which is what puts
/// several of DCluster's degree classes to work at once.
pub fn preferential_attachment(n: usize, k: usize, seed: u64) -> Graph {
    assert!(n > k && k >= 1);
    let mut rng = SplitMix64::seed(seed);
    let m0 = k + 1;
    let mut edges = Vec::new();
    let mut targets: Vec<u32> = Vec::new();
    for u in 0..m0 {
        for v in (u + 1)..m0 {
            edges.push((u as u32, v as u32));
            targets.push(u as u32);
            targets.push(v as u32);
        }
    }
    for v in m0..n {
        let mut picked: Vec<u32> = Vec::with_capacity(k);
        let mut guard = 0;
        while picked.len() < k && guard < 100 * k {
            guard += 1;
            let cand = targets[rng.below(targets.len() as u64) as usize];
            if !picked.contains(&cand) {
                picked.push(cand);
            }
        }
        for &t in &picked {
            edges.push((t, v as u32));
            targets.push(t);
            targets.push(v as u32);
        }
    }
    Graph::from_edges(n, &edges)
}

/// Path on `n` vertices.
pub fn path(n: usize) -> Graph {
    let edges: Vec<(u32, u32)> = (0..n.saturating_sub(1)).map(|u| (u as u32, u as u32 + 1)).collect();
    Graph::from_edges(n, &edges)
}

/// Cycle on `n` vertices.
pub fn cycle(n: usize) -> Graph {
    let mut edges: Vec<(u32, u32)> = (0..n.saturating_sub(1)).map(|u| (u as u32, u as u32 + 1)).collect();
    if n > 2 {
        edges.push((n as u32 - 1, 0));
    }
    Graph::from_edges(n, &edges)
}

/// Star with vertex 0 at the center.
pub fn star(n: usize) -> Graph {
    let edges: Vec<(u32, u32)> = (1..n).map(|v| (0u32, v as u32)).collect();
    Graph::from_edges(n, &edges)
}

/// Complete graph.
pub fn clique(n: usize) -> Graph {
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            edges.push((u as u32, v as u32));
        }
    }
    Graph::from_edges(n, &edges)
}

/// Two cliques of `n / 2` vertices joined by a single edge. The diameter is
/// large relative to the degrees, which is the shape the paper cares about.
pub fn barbell(n: usize) -> Graph {
    let half = n / 2;
    let mut edges = Vec::new();
    for u in 0..half {
        for v in (u + 1)..half {
            edges.push((u as u32, v as u32));
        }
    }
    for u in half..n {
        for v in (u + 1)..n {
            edges.push((u as u32, v as u32));
        }
    }
    if half > 0 && half < n {
        edges.push((0, half as u32));
    }
    Graph::from_edges(n, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_deduped_and_symmetric() {
        let g = Graph::from_edges(4, &[(0, 1), (1, 0), (0, 1), (2, 2), (2, 3)]);
        assert_eq!(g.m(), 2);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(2), 1);
        assert!(g.is_neighbor(1, 0) && g.is_neighbor(0, 1));
        assert!(!g.is_neighbor(0, 3));
        assert_eq!(g.edges(), vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn generators_have_the_expected_shape() {
        assert_eq!(path(10).m(), 9);
        assert_eq!(cycle(10).m(), 10);
        assert_eq!(star(10).m(), 9);
        assert_eq!(star(10).degree(0), 9);
        assert_eq!(clique(10).m(), 45);
        assert_eq!(barbell(10).m(), 21);
        let pa = preferential_attachment(64, 2, 5);
        assert_eq!(pa.n(), 64);
        let max_deg = (0..64).map(|v| pa.degree(v)).max().unwrap();
        let min_deg = (0..64).map(|v| pa.degree(v)).min().unwrap();
        assert!(max_deg > 4 * min_deg, "expected a skewed degree sequence");
    }

    #[test]
    fn generators_are_deterministic() {
        assert_eq!(erdos_renyi(50, 0.2, 9).edges(), erdos_renyi(50, 0.2, 9).edges());
        assert_eq!(
            preferential_attachment(50, 3, 9).edges(),
            preferential_attachment(50, 3, 9).edges()
        );
    }
}
