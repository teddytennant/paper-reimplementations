//! Tree decompositions of the underlying undirected graph, width at most 2.

use crate::graph::Graph;
use std::collections::VecDeque;

/// A tree decomposition: bags of original vertices and tree edges on bag indices.
///
/// Callers may supply an arbitrary decomposition of width ≤ 2. Helpers build
/// one for paths, trees (width 1), and 2-trees / series-parallel bags of size 3.
#[derive(Clone, Debug)]
pub struct TreeDecomposition {
    pub bags: Vec<Vec<usize>>,
    pub tree_edges: Vec<(usize, usize)>,
}

impl TreeDecomposition {
    pub fn new(bags: Vec<Vec<usize>>, tree_edges: Vec<(usize, usize)>) -> Self {
        let mut td = Self { bags, tree_edges };
        td.normalize();
        td
    }

    fn normalize(&mut self) {
        for bag in &mut self.bags {
            bag.sort_unstable();
            bag.dedup();
        }
    }

    /// Width = max bag size − 1.
    pub fn width(&self) -> usize {
        self.bags
            .iter()
            .map(|b| b.len())
            .max()
            .unwrap_or(0)
            .saturating_sub(1)
    }

    /// Path on `n` vertices: bags `{i, i+1}` linked in a path.
    pub fn path(n: usize) -> Self {
        if n == 0 {
            return Self {
                bags: vec![],
                tree_edges: vec![],
            };
        }
        if n == 1 {
            return Self {
                bags: vec![vec![0]],
                tree_edges: vec![],
            };
        }
        let bags: Vec<Vec<usize>> = (0..n - 1).map(|i| vec![i, i + 1]).collect();
        let tree_edges: Vec<(usize, usize)> =
            (0..n.saturating_sub(2)).map(|i| (i, i + 1)).collect();
        Self { bags, tree_edges }
    }

    /// Width-1 decomposition of an undirected tree (or forest) on `n` vertices.
    pub fn for_tree(n: usize, undirected_edges: &[(usize, usize)]) -> Self {
        if n == 0 {
            return Self {
                bags: vec![],
                tree_edges: vec![],
            };
        }
        if n == 1 {
            return Self {
                bags: vec![vec![0]],
                tree_edges: vec![],
            };
        }

        let mut adj = vec![vec![]; n];
        for &(u, v) in undirected_edges {
            adj[u].push(v);
            adj[v].push(u);
        }

        let mut parent = vec![None; n];
        let mut seen = vec![false; n];
        let mut component_roots = Vec::new();
        for r in 0..n {
            if seen[r] {
                continue;
            }
            component_roots.push(r);
            seen[r] = true;
            let mut stack = vec![r];
            while let Some(u) = stack.pop() {
                for &v in &adj[u] {
                    if !seen[v] {
                        seen[v] = true;
                        parent[v] = Some(u);
                        stack.push(v);
                    }
                }
            }
        }

        let mut bags: Vec<Vec<usize>> = Vec::new();
        let mut vertex_bag = vec![usize::MAX; n];
        for v in 0..n {
            if let Some(p) = parent[v] {
                vertex_bag[v] = bags.len();
                bags.push(vec![p.min(v), p.max(v)]);
            }
        }

        if bags.is_empty() {
            let bags: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
            let tree_edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
            return Self { bags, tree_edges };
        }

        let mut tree_edges: Vec<(usize, usize)> = Vec::new();
        for v in 0..n {
            if vertex_bag[v] == usize::MAX {
                continue;
            }
            let p = parent[v].unwrap();
            if vertex_bag[p] != usize::MAX {
                tree_edges.push((vertex_bag[v], vertex_bag[p]));
            }
        }

        for &r in &component_roots {
            let child_bags: Vec<usize> = (0..n)
                .filter(|&v| parent[v] == Some(r) && vertex_bag[v] != usize::MAX)
                .map(|v| vertex_bag[v])
                .collect();
            for i in 1..child_bags.len() {
                tree_edges.push((child_bags[0], child_bags[i]));
            }
        }

        let mut first_bag: Vec<usize> = Vec::new();
        for &r in &component_roots {
            let child = (0..n).find(|&v| parent[v] == Some(r) && vertex_bag[v] != usize::MAX);
            if let Some(v) = child {
                first_bag.push(vertex_bag[v]);
            } else {
                let id = bags.len();
                bags.push(vec![r]);
                first_bag.push(id);
            }
        }
        for i in 1..first_bag.len() {
            tree_edges.push((first_bag[0], first_bag[i]));
        }

        let mut td = Self { bags, tree_edges };
        td.normalize();
        td
    }

    /// Width-2 decomposition whose bags are the given triples (2-tree / SP).
    ///
    /// Dual bags that share two vertices are joined by a spanning tree.
    pub fn for_width2(triangles: &[[usize; 3]]) -> Self {
        let m = triangles.len();
        if m == 0 {
            return Self {
                bags: vec![],
                tree_edges: vec![],
            };
        }
        let bags: Vec<Vec<usize>> = triangles
            .iter()
            .map(|t| {
                let mut v = vec![t[0], t[1], t[2]];
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        if m == 1 {
            return Self {
                bags,
                tree_edges: vec![],
            };
        }

        let mut dual = vec![vec![]; m];
        for i in 0..m {
            for j in i + 1..m {
                let shared = bags[i].iter().filter(|v| bags[j].contains(v)).count();
                if shared >= 2 {
                    dual[i].push(j);
                    dual[j].push(i);
                }
            }
        }

        let mut tree_edges = Vec::new();
        let mut seen = vec![false; m];
        let mut q = VecDeque::new();
        for start in 0..m {
            if seen[start] {
                continue;
            }
            if start > 0 {
                tree_edges.push((0, start));
            }
            seen[start] = true;
            q.push_back(start);
            while let Some(u) = q.pop_front() {
                for &v in &dual[u] {
                    if !seen[v] {
                        seen[v] = true;
                        tree_edges.push((u, v));
                        q.push_back(v);
                    }
                }
            }
        }
        Self { bags, tree_edges }
    }

    /// 4-cycle `0-1-2-3-0` with chord `0-2`: bags `{0,1,2}` and `{0,2,3}`.
    pub fn cycle4_with_chord() -> Self {
        Self::for_width2(&[[0, 1, 2], [0, 2, 3]])
    }

    pub fn bag_adj(&self) -> Vec<Vec<usize>> {
        let m = self.bags.len();
        let mut adj = vec![vec![]; m];
        for &(u, v) in &self.tree_edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }

    /// Check vertex/edge coverage, tree shape, running intersection, width ≤ 2.
    pub fn validate(&self, graph: &Graph) -> Result<(), String> {
        if self.bags.is_empty() {
            if graph.n() == 0 {
                return Ok(());
            }
            return Err("empty decomposition".into());
        }
        if self.bags.iter().any(|b| b.len() > 3) {
            return Err("only width ≤ 2 is implemented (bags of size ≤ 3)".into());
        }
        let m = self.bags.len();
        let adj = self.bag_adj();
        let mut seen = vec![false; m];
        let mut stack = vec![0];
        seen[0] = true;
        let mut visited = 0;
        while let Some(u) = stack.pop() {
            visited += 1;
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        if visited != m {
            return Err("decomposition is not a tree (disconnected)".into());
        }
        if self.tree_edges.len() != m.saturating_sub(1) {
            return Err("decomposition is not a tree (wrong number of edges)".into());
        }

        let mut covered = vec![false; graph.n()];
        for bag in &self.bags {
            for &v in bag {
                if v >= graph.n() {
                    return Err(format!("bag vertex {v} out of range"));
                }
                covered[v] = true;
            }
        }
        if covered.iter().any(|&c| !c) {
            return Err("some vertices are not in any bag".into());
        }

        for e in graph.edges() {
            let ok = self
                .bags
                .iter()
                .any(|b| b.contains(&e.from) && b.contains(&e.to));
            if !ok {
                return Err(format!("edge {}->{} is not covered by a bag", e.from, e.to));
            }
        }

        for v in 0..graph.n() {
            let idxs: Vec<usize> = (0..m).filter(|&i| self.bags[i].contains(&v)).collect();
            if idxs.is_empty() {
                continue;
            }
            let mut sub_seen = vec![false; m];
            let mut st = vec![idxs[0]];
            sub_seen[idxs[0]] = true;
            let mut count = 0;
            while let Some(u) = st.pop() {
                count += 1;
                for &w in &adj[u] {
                    if !sub_seen[w] && self.bags[w].contains(&v) {
                        sub_seen[w] = true;
                        st.push(w);
                    }
                }
            }
            if count != idxs.len() {
                return Err(format!("running intersection fails at vertex {v}"));
            }
        }
        Ok(())
    }
}
