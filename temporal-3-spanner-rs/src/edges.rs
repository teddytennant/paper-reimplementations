//! Accumulator for a set of biclique edges.

/// A set of `(source, target)` pairs, held as a bitmap over the full biclique.
///
/// The recursion unions edge sets from five places, so dedup has to be cheap and
/// the iteration order has to be fixed for the output to be reproducible.
#[derive(Clone, Debug)]
pub struct EdgeSet {
    n_sources: usize,
    n_targets: usize,
    mask: Vec<bool>,
    len: usize,
}

impl EdgeSet {
    pub fn new(n_sources: usize, n_targets: usize) -> Self {
        EdgeSet {
            n_sources,
            n_targets,
            mask: vec![false; n_sources * n_targets],
            len: 0,
        }
    }

    #[inline]
    pub fn insert(&mut self, s: usize, t: usize) {
        let i = s * self.n_targets + t;
        if !self.mask[i] {
            self.mask[i] = true;
            self.len += 1;
        }
    }

    #[inline]
    pub fn contains(&self, s: usize, t: usize) -> bool {
        self.mask[s * self.n_targets + t]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn union_with(&mut self, other: &EdgeSet) {
        assert_eq!(self.mask.len(), other.mask.len());
        for i in 0..self.mask.len() {
            if other.mask[i] && !self.mask[i] {
                self.mask[i] = true;
                self.len += 1;
            }
        }
    }

    /// The edges in lexicographic order.
    pub fn to_vec(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.len);
        for s in 0..self.n_sources {
            for t in 0..self.n_targets {
                if self.mask[s * self.n_targets + t] {
                    out.push((s, t));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_is_idempotent_and_output_is_sorted() {
        let mut e = EdgeSet::new(3, 3);
        assert!(e.is_empty());
        e.insert(2, 1);
        e.insert(0, 2);
        e.insert(2, 1);
        assert_eq!(e.len(), 2);
        assert!(e.contains(2, 1));
        assert!(!e.contains(1, 2));
        assert_eq!(e.to_vec(), vec![(0, 2), (2, 1)]);
    }

    #[test]
    fn union_counts_each_edge_once() {
        let mut a = EdgeSet::new(2, 2);
        a.insert(0, 0);
        a.insert(1, 1);
        let mut b = EdgeSet::new(2, 2);
        b.insert(1, 1);
        b.insert(0, 1);
        a.union_with(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.to_vec(), vec![(0, 0), (0, 1), (1, 1)]);
    }
}
