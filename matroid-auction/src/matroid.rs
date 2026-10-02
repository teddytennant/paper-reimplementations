use std::cell::Cell;

/// Independence oracle for a matroid on a finite ground set.
///
/// The auction calls only this method. It does not inspect the matroid type.
pub trait IndependenceOracle {
    fn independent(&self, set: &[usize]) -> bool;
}

fn unique_sorted(set: &[usize]) -> Vec<usize> {
    let mut v = set.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// Uniform matroid: a set is independent iff its cardinality is at most `k`.
pub struct UniformMatroid {
    k: usize,
    queries: Cell<u64>,
}

impl UniformMatroid {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            queries: Cell::new(0),
        }
    }

    pub fn queries(&self) -> u64 {
        self.queries.get()
    }

    pub fn reset_queries(&self) {
        self.queries.set(0);
    }
}

impl IndependenceOracle for UniformMatroid {
    fn independent(&self, set: &[usize]) -> bool {
        self.queries.set(self.queries.get() + 1);
        unique_sorted(set).len() <= self.k
    }
}

/// Partition matroid given as a list of (group, capacity).
///
/// Each element belongs to at most one group. Elements omitted from every
/// group are free (their own group of capacity 1).
pub struct PartitionMatroid {
    group_of: Vec<usize>,
    capacity: Vec<usize>,
    queries: Cell<u64>,
}

impl PartitionMatroid {
    pub fn new(n: usize, groups: &[(Vec<usize>, usize)]) -> Self {
        let mut group_of = vec![usize::MAX; n];
        let mut capacity = Vec::new();
        for (elems, cap) in groups {
            let g = capacity.len();
            capacity.push(*cap);
            for &e in elems {
                assert!(e < n, "partition element {e} out of range");
                assert!(
                    group_of[e] == usize::MAX,
                    "element {e} assigned to two groups"
                );
                group_of[e] = g;
            }
        }
        for e in 0..n {
            if group_of[e] == usize::MAX {
                let g = capacity.len();
                capacity.push(1);
                group_of[e] = g;
            }
        }
        Self {
            group_of,
            capacity,
            queries: Cell::new(0),
        }
    }

    pub fn queries(&self) -> u64 {
        self.queries.get()
    }

    pub fn reset_queries(&self) {
        self.queries.set(0);
    }
}

impl IndependenceOracle for PartitionMatroid {
    fn independent(&self, set: &[usize]) -> bool {
        self.queries.set(self.queries.get() + 1);
        let mut counts = vec![0usize; self.capacity.len()];
        for &e in &unique_sorted(set) {
            if e >= self.group_of.len() {
                return false;
            }
            let g = self.group_of[e];
            counts[g] += 1;
            if counts[g] > self.capacity[g] {
                return false;
            }
        }
        true
    }
}

/// Transversal matroid. Left vertices are the ground set. A set is
/// independent iff it has a matching into the right side.
pub struct TransversalMatroid {
    n_right: usize,
    adj: Vec<Vec<usize>>,
    queries: Cell<u64>,
}

impl TransversalMatroid {
    pub fn new(n_right: usize, adj: Vec<Vec<usize>>) -> Self {
        for nbrs in &adj {
            for &v in nbrs {
                assert!(v < n_right, "right vertex {v} out of range");
            }
        }
        Self {
            n_right,
            adj,
            queries: Cell::new(0),
        }
    }

    pub fn queries(&self) -> u64 {
        self.queries.get()
    }

    pub fn reset_queries(&self) {
        self.queries.set(0);
    }

    fn dfs(&self, left: usize, owner: &mut [Option<usize>], seen: &mut [bool]) -> bool {
        for &right in &self.adj[left] {
            if seen[right] {
                continue;
            }
            seen[right] = true;
            if owner[right].is_none() || self.dfs(owner[right].unwrap(), owner, seen) {
                owner[right] = Some(left);
                return true;
            }
        }
        false
    }
}

impl IndependenceOracle for TransversalMatroid {
    fn independent(&self, set: &[usize]) -> bool {
        self.queries.set(self.queries.get() + 1);
        let elems = unique_sorted(set);
        let mut owner = vec![None; self.n_right];
        for &left in &elems {
            if left >= self.adj.len() {
                return false;
            }
            let mut seen = vec![false; self.n_right];
            if !self.dfs(left, &mut owner, &mut seen) {
                return false;
            }
        }
        true
    }
}
