//! Directed graphs: adjacency lists, sources/sinks, SCCs, and strong connectivity.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiGraph {
    n: usize,
    adj: Vec<Vec<usize>>,
    adj_rev: Vec<Vec<usize>>,
}

impl DiGraph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            adj_rev: vec![Vec::new(); n],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn out(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    pub fn incoming(&self, v: usize) -> &[usize] {
        &self.adj_rev[v]
    }

    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.adj[u].contains(&v)
    }

    /// Adds a simple directed edge. Self-loops and duplicates are ignored.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n, "vertex out of range");
        if u == v || self.adj[u].contains(&v) {
            return;
        }
        self.adj[u].push(v);
        self.adj_rev[v].push(u);
    }

    pub fn with_edges(&self, extra: &[(usize, usize)]) -> Self {
        let mut g = self.clone();
        for &(u, v) in extra {
            g.add_edge(u, v);
        }
        g
    }

    /// Vertices of indegree zero. Isolated vertices are sources.
    pub fn sources(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&v| self.adj_rev[v].is_empty())
            .collect()
    }

    /// Vertices of outdegree zero. Isolated vertices are sinks.
    pub fn sinks(&self) -> Vec<usize> {
        (0..self.n).filter(|&v| self.adj[v].is_empty()).collect()
    }

    /// Terminals Z = S union T, sorted.
    pub fn terminals(&self) -> Vec<usize> {
        let mut z = self.sources();
        for v in self.sinks() {
            if !z.contains(&v) {
                z.push(v);
            }
        }
        z.sort_unstable();
        z
    }

    pub fn reachable_from(&self, src: usize) -> Vec<bool> {
        let mut seen = vec![false; self.n];
        if src >= self.n {
            return seen;
        }
        let mut stack = vec![src];
        seen[src] = true;
        while let Some(v) = stack.pop() {
            for &w in &self.adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        seen
    }

    pub fn is_dag(&self) -> bool {
        let mut indeg: Vec<usize> = (0..self.n).map(|v| self.adj_rev[v].len()).collect();
        let mut q: Vec<usize> = (0..self.n).filter(|&v| indeg[v] == 0).collect();
        let mut seen = 0;
        while let Some(v) = q.pop() {
            seen += 1;
            for &w in &self.adj[v] {
                indeg[w] -= 1;
                if indeg[w] == 0 {
                    q.push(w);
                }
            }
        }
        seen == self.n
    }

    /// Kosaraju strongly connected components.
    pub fn sccs(&self) -> Vec<Vec<usize>> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }
        let mut seen = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for v in 0..n {
            if !seen[v] {
                dfs_finish(v, &self.adj, &mut seen, &mut order);
            }
        }
        let mut assigned = vec![false; n];
        let mut comps = Vec::new();
        for &v in order.iter().rev() {
            if !assigned[v] {
                let mut comp = Vec::new();
                dfs_finish(v, &self.adj_rev, &mut assigned, &mut comp);
                comps.push(comp);
            }
        }
        comps
    }

    pub fn is_strongly_connected(&self) -> bool {
        if self.n <= 1 {
            return true;
        }
        self.sccs().len() == 1
    }

    /// Condensation DAG of SCCs, plus the SCC index of each original vertex.
    pub fn condensation(&self) -> (DiGraph, Vec<usize>) {
        let comps = self.sccs();
        let mut id = vec![0; self.n];
        for (i, comp) in comps.iter().enumerate() {
            for &v in comp {
                id[v] = i;
            }
        }
        let mut dag = DiGraph::new(comps.len());
        for u in 0..self.n {
            for &v in &self.adj[u] {
                if id[u] != id[v] {
                    dag.add_edge(id[u], id[v]);
                }
            }
        }
        (dag, id)
    }
}

fn dfs_finish(v: usize, adj: &[Vec<usize>], seen: &mut [bool], order: &mut Vec<usize>) {
    seen[v] = true;
    for &w in &adj[v] {
        if !seen[w] {
            dfs_finish(w, adj, seen, order);
        }
    }
    order.push(v);
}
