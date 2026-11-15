//! Graphicality tests and heap-based Havel–Hakimi constructions.
//!
//! Intra-cell: Erdős–Gállai and heap-based Havel–Hakimi (Algorithm 1, diagonal
//! layers). Inter-cell: Gale–Ryser and heap-based bipartite Havel–Hakimi
//! (Algorithm 1, off-diagonal layers).

use std::collections::BinaryHeap;

use crate::graph::Error;

/// A `d`-regular sequence of length `n` is graphic iff `0 ≤ d ≤ n-1` (for
/// `n ≥ 1`) and `n d` is even. The `n = 0` sequence is graphic only for `d = 0`.
pub fn is_graphic_regular(n: usize, d: usize) -> bool {
    if n == 0 {
        return d == 0;
    }
    d <= n - 1 && (n * d) % 2 == 0
}

/// Erdős–Gállai: a nonincreasing sequence `d` is graphic iff its sum is even
/// and for every `k = 1..n`,
/// `∑_{i=1}^k d_i ≤ k(k-1) + ∑_{i=k+1}^n min(k, d_i)`.
pub fn erdos_gallai(degrees: &[usize]) -> bool {
    let n = degrees.len();
    let mut d = degrees.to_vec();
    d.sort_unstable_by(|a, b| b.cmp(a));
    let sum: usize = d.iter().sum();
    if sum % 2 == 1 {
        return false;
    }
    if n == 0 {
        return true;
    }
    if d[0] > n - 1 {
        return false;
    }
    let mut prefix = vec![0usize; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + d[i];
    }
    for k in 1..=n {
        let left = prefix[k];
        let mut right = k * (k - 1);
        for &di in d.iter().skip(k) {
            right += di.min(k);
        }
        if left > right {
            return false;
        }
    }
    true
}

/// Gale–Ryser: bipartite graphicness of `(a, b)`.
///
/// Sort `a1 ≥ ⋯ ≥ am`, `b1 ≥ ⋯ ≥ bn`. The pair is bipartite-graphic iff
/// `∑ a = ∑ b` and for every `k = 1..m`,
/// `∑_{t=1}^k a_t ≤ ∑_s min(k, b_s)`. Each `a_i` must also be at most `n`
/// and each `b_j` at most `m`.
pub fn gale_ryser(left_deg: &[usize], right_deg: &[usize]) -> bool {
    let mut a = left_deg.to_vec();
    let mut b = right_deg.to_vec();
    a.sort_unstable_by(|x, y| y.cmp(x));
    b.sort_unstable_by(|x, y| y.cmp(x));
    let sum_a: usize = a.iter().sum();
    let sum_b: usize = b.iter().sum();
    if sum_a != sum_b {
        return false;
    }
    let m = a.len();
    let n = b.len();
    if a.iter().any(|&x| x > n) {
        return false;
    }
    if b.iter().any(|&x| x > m) {
        return false;
    }
    let mut prefix = 0usize;
    for k in 1..=m {
        prefix += a[k - 1];
        let rhs: usize = b.iter().map(|&bs| bs.min(k)).sum();
        if prefix > rhs {
            return false;
        }
    }
    true
}

/// Heap-based Havel–Hakimi for a simple undirected graph.
///
/// Binary max-heap of remaining degrees. The vertex of currently largest
/// remaining degree `d` is connected to the `d` other vertices of largest
/// remaining degree; those degrees are decremented and pushed back. Returns
/// undirected edges `(u, v)` with `u < v` on local indices `0..n`.
pub fn havel_hakimi(degrees: &[usize]) -> Result<Vec<(usize, usize)>, Error> {
    let n = degrees.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if degrees.iter().any(|&d| d > n - 1) {
        return Err(Error::InvalidDegree);
    }
    if !erdos_gallai(degrees) {
        return Err(Error::NotGraphic);
    }
    // Max-heap of (remaining degree, vertex). A vertex sits in the heap at
    // most once: it is popped either as the source or as a neighbour and, in
    // the neighbour case, re-inserted with the decremented degree.
    let mut heap: BinaryHeap<(usize, usize)> = degrees
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > 0)
        .map(|(i, d)| (*d, i))
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    while let Some((d, u)) = heap.pop() {
        if d == 0 {
            break;
        }
        let mut nbrs: Vec<(usize, usize)> = Vec::with_capacity(d);
        for _ in 0..d {
            match heap.pop() {
                Some(entry) => nbrs.push(entry),
                None => return Err(Error::NotGraphic),
            }
        }
        for (nd, v) in nbrs {
            if v == u || nd == 0 {
                return Err(Error::NotGraphic);
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            edges.push((a, b));
            let new_d = nd - 1;
            if new_d > 0 {
                heap.push((new_d, v));
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    Ok(edges)
}

/// Heap-based bipartite Havel–Hakimi / configuration.
///
/// Each left vertex is connected to the current highest-remaining-degree right
/// vertices. Returns edges `(left_index, right_index)`.
pub fn bipartite_havel_hakimi(
    left: &[usize],
    right: &[usize],
) -> Result<Vec<(usize, usize)>, Error> {
    if !gale_ryser(left, right) {
        return Err(Error::NotBipartiteGraphic);
    }
    let mut right_heap: BinaryHeap<(usize, usize)> = right
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > 0)
        .map(|(i, d)| (*d, i))
        .collect();
    let mut left_heap: BinaryHeap<(usize, usize)> = left
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > 0)
        .map(|(i, d)| (*d, i))
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    while let Some((d, u)) = left_heap.pop() {
        if d == 0 {
            continue;
        }
        let mut nbrs: Vec<(usize, usize)> = Vec::with_capacity(d);
        for _ in 0..d {
            match right_heap.pop() {
                Some(entry) => nbrs.push(entry),
                None => return Err(Error::NotBipartiteGraphic),
            }
        }
        for (nd, v) in nbrs {
            if nd == 0 {
                return Err(Error::NotBipartiteGraphic);
            }
            edges.push((u, v));
            let new_d = nd - 1;
            if new_d > 0 {
                right_heap.push((new_d, v));
            }
        }
    }
    if right_heap.iter().any(|&(d, _)| d > 0) {
        return Err(Error::NotBipartiteGraphic);
    }
    edges.sort_unstable();
    Ok(edges)
}
