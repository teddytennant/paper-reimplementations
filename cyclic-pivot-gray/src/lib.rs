//! Cyclic pivot Gray codes for well-ordered k-degenerate graphs.
//!
//! Algorithms follow arXiv:2610.09410 (Algorithms 1-6). Ranks are 1-based,
//! matching the paper's ranking procedures.

use num_bigint::BigUint;
use num_traits::{One, Zero};

/// Number of binary strings of length `n` with weight at most `k`.
pub fn subset_count(n: usize, k: usize) -> BigUint {
    let k = k.min(n);
    let mut c = BigUint::one();
    let mut total = BigUint::one();
    for i in 1..=k {
        // C(n, i) = C(n, i-1) * (n-i+1) / i
        c *= n - i + 1;
        c /= i;
        total += &c;
    }
    total
}

/// Binomial coefficient C(n, k).
pub fn binom(n: usize, k: usize) -> BigUint {
    if k > n {
        return BigUint::zero();
    }
    let k = k.min(n - k);
    let mut c = BigUint::one();
    for i in 1..=k {
        c *= n - k + i;
        c /= i;
    }
    c
}

/// Number of well-ordered k-degenerate graphs on n labeled vertices.
///
/// λ(n, k) = prod_{i=1}^{n-1} (sum_{j=0}^{min(k, i)} C(i, j)).
pub fn lambda(n: usize, k: usize) -> BigUint {
    if n == 0 {
        return BigUint::zero();
    }
    let mut acc = BigUint::one();
    for i in 1..n {
        acc *= subset_count(i, k);
    }
    acc
}

/// Suffix-weight table W[j][t] = number of length-(n-j) strings of weight ≤ t.
///
/// Algorithm 3, lines 2-6. W is 1-indexed in the paper; here W[j][t] is the
/// same quantity, with j in 1..=n and t in 0..=k. W[n][*] = 1.
pub fn suffix_table(n: usize, k: usize) -> Vec<Vec<BigUint>> {
    let mut w = vec![vec![BigUint::one(); k + 1]; n + 1];
    if n == 0 {
        return w;
    }
    for j in (1..n).rev() {
        let len = n - j;
        for t in 0..=k {
            let mut s = BigUint::zero();
            let tmax = t.min(len);
            let mut c = BigUint::one();
            s += &c;
            for i in 1..=tmax {
                c *= len - i + 1;
                c /= i;
                s += &c;
            }
            w[j][t] = s;
        }
    }
    w
}

/// Number of length-`m` strings of weight at most `budget`, Appendix B.
///
/// `count(m, K) = sum_{i=0}^{min(m,K)} C(m, i)`, with `count(m, K<0) = 0`.
pub fn count_le(m: usize, budget: isize) -> BigUint {
    if budget < 0 {
        return BigUint::zero();
    }
    let cap = budget as usize;
    (0..=m.min(cap)).map(|i| binom(m, i)).sum()
}

/// Rank a binary string in the pruned BRGC for weight ≤ k. 1-based.
///
/// Appendix B. Bits are visited from the last to the first. `d` starts at 1
/// and tracks the running parity of bits already decided (plus one).
pub fn ranking(b: &[u8], k: usize) -> BigUint {
    let mut n = b.len();
    let mut k = k as isize;
    let mut r = BigUint::one();
    let mut d: usize = 1;
    for &bit in b.iter().rev() {
        n -= 1;
        let s_i = bit as isize;
        if s_i == (d % 2) as isize {
            r += count_le(n, k - 1 + s_i);
        }
        d += bit as usize;
        k -= s_i;
    }
    r
}

/// Rank using an explicit suffix table so the paper's W[j][k-weight] is exact.
#[allow(dead_code)]
fn ranking_with(b: &[u8], k: usize, w: &[Vec<BigUint>]) -> BigUint {
    let n = b.len();
    let mut r = BigUint::one();
    let mut parity: u8 = 0;
    let mut weight: usize = 0;
    for j in 1..=n {
        let bit = b[j - 1];
        let budget = k - weight;
        if parity == 0 {
            if bit == 1 {
                r += &w[j][budget];
                parity ^= 1;
                weight += 1;
            }
        } else if bit == 0 {
            r += &w[j][budget];
            parity ^= 1;
        } else {
            parity ^= 1;
            weight += 1;
        }
    }
    r
}

/// Unrank `x` (1-based) in the pruned BRGC of length `n`, weight ≤ `k`.
///
/// Appendix B. Bits are decided from the last to the first.
pub fn unranking(n: usize, k: usize, x: &BigUint) -> Vec<u8> {
    let mut b = vec![0u8; n];
    let mut k = k as isize;
    let mut d: usize = 1;
    let mut r = x - BigUint::one();
    for i in (0..n).rev() {
        let threshold = count_le(i, k - 1 + (d % 2) as isize);
        if r >= threshold {
            b[i] = (d % 2) as u8;
            r -= threshold;
        } else {
            b[i] = 1 - (d % 2) as u8;
        }
        k -= b[i] as isize;
        d += b[i] as usize;
    }
    b
}

/// Reflect-and-prefix list of all weight-≤k strings of length n.
///
/// Algorithm 1. Direction +1 is the list as built (first string is 0^n).
/// Direction -1 is that list reversed.
pub fn pruned_brgc(n: usize, k: usize) -> Vec<Vec<u8>> {
    fn rec(n: usize, k: usize) -> Vec<Vec<u8>> {
        if k == 0 {
            return vec![vec![0u8; n]];
        }
        if n == 1 {
            // k ≥ 1: 0 then 1
            return vec![vec![0], vec![1]];
        }
        let mut out = Vec::new();
        for mut s in rec(n - 1, k) {
            s.push(0);
            out.push(s);
        }
        let mut hi = rec(n - 1, k - 1);
        hi.reverse();
        for mut s in hi {
            s.push(1);
            out.push(s);
        }
        out
    }
    rec(n, k)
}

/// List `B_{n,k}` in the direction used by GenWOD.
///
/// `forward = true` is the Algorithm 1 order (starts at 0^n).
pub fn list_subsets(n: usize, k: usize, forward: bool) -> Vec<Vec<u8>> {
    let mut xs = pruned_brgc(n, k);
    if !forward {
        xs.reverse();
    }
    xs
}

/// A well-ordered k-degenerate graph as a strict lower-triangular adjacency
/// list: `edges[i]` is the neighbor set of vertex `i+1` among {1, ..., i},
/// stored as a bit vector of length i. Vertex labels are 1-based in the paper;
/// bit j (0-based) of row i corresponds to vertex j+1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Graph {
    pub n: usize,
    /// rows[v] has length v, for v = 0..n-1 (vertex v+1).
    pub rows: Vec<Vec<u8>>,
}

impl Graph {
    pub fn empty(n: usize) -> Self {
        Self {
            n,
            rows: (0..n).map(|v| vec![0u8; v]).collect(),
        }
    }

    pub fn edge_set(&self) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for (v, row) in self.rows.iter().enumerate() {
            for (u, bit) in row.iter().enumerate() {
                if *bit == 1 {
                    e.push((u + 1, v + 1));
                }
            }
        }
        e
    }
}

/// Generate all well-ordered k-degenerate graphs in cyclic pivot Gray code order.
///
/// Algorithm 2. The first graph is the empty graph.
pub fn gen_wod(n: usize, k: usize) -> Vec<Graph> {
    fn rec(n: usize, k: usize) -> Vec<Graph> {
        if n == 1 {
            return vec![Graph::empty(1)];
        }
        if k == 0 {
            return vec![Graph::empty(n)];
        }
        let bases = rec(n - 1, k);
        let mut out = Vec::new();
        // d = 0 on the first base: direction (+1), i.e. forward.
        for (i, base) in bases.into_iter().enumerate() {
            let forward = i % 2 == 0;
            for s in list_subsets(n - 1, k, forward) {
                let mut g = base.clone();
                g.n = n;
                g.rows.push(s);
                out.push(g);
            }
        }
        out
    }
    rec(n, k)
}

/// Rank a well-ordered k-degenerate graph. 1-based. Algorithm 5.
pub fn graph_ranking(g: &Graph, k: usize) -> BigUint {
    fn rec(rows: &[Vec<u8>], k: usize) -> BigUint {
        let n = rows.len();
        if n <= 1 {
            return BigUint::one();
        }
        let i = rec(&rows[..n - 1], k);
        let b = &rows[n - 1];
        let m = subset_count(n - 1, k);
        let j = if (&i % 2u32) == BigUint::one() {
            ranking(b, k)
        } else {
            &m - &ranking(b, k) + BigUint::one()
        };
        (&i - BigUint::one()) * &m + j
    }
    rec(&g.rows, k)
}

/// Unrank `x` (1-based) to the graph of that rank. Algorithm 6.
pub fn graph_unranking(x: &BigUint, n: usize, k: usize) -> Graph {
    fn rec(x: &BigUint, n: usize, k: usize) -> Graph {
        if n <= 1 {
            return Graph::empty(1);
        }
        let m = subset_count(n - 1, k);
        let i = (x - BigUint::one()) / &m + BigUint::one();
        let j = x - (&i - BigUint::one()) * &m;
        let mut g = rec(&i, n - 1, k);
        let b = if (&i % 2u32) == BigUint::one() {
            unranking(n - 1, k, &j)
        } else {
            let rev = &m - &j + BigUint::one();
            unranking(n - 1, k, &rev)
        };
        g.n = n;
        g.rows.push(b);
        g
    }
    rec(x, n, k)
}

/// Symmetric difference of the edge sets, as unordered pairs.
pub fn edge_symdiff(a: &Graph, b: &Graph) -> Vec<(usize, usize)> {
    let ea = a.edge_set();
    let eb = b.edge_set();
    let mut out = Vec::new();
    for e in &ea {
        if !eb.contains(e) {
            out.push(*e);
        }
    }
    for e in &eb {
        if !ea.contains(e) {
            out.push(*e);
        }
    }
    out
}

/// A transition is a pivot when the two graphs differ by deleting one edge
/// and adding another that shares a vertex.
pub fn is_pivot(diff: &[(usize, usize)]) -> bool {
    if diff.len() != 2 {
        return false;
    }
    let (a, b) = diff[0];
    let (c, d) = diff[1];
    a == c || a == d || b == c || b == d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_8_n4_k2() {
        let order = pruned_brgc(4, 2);
        let as_str: Vec<String> = order
            .iter()
            .map(|b| b.iter().map(|x| char::from(b'0' + x)).collect())
            .collect();
        // The listing printed by Appendix A (and by the generation tree in
        // Section 2). The comma-separated line a few paragraphs later drops
        // 1010 and reorders the tail; both cannot be the BRGC, and the
        // recursion decides it.
        assert_eq!(
            as_str,
            vec![
                "0000", "1000", "1100", "0100", "0110", "1010", "0010", "0011", "0101", "1001",
                "0001"
            ]
        );
    }

    #[test]
    fn example_11_ranking() {
        // 0110 has rank 5
        let b = [0u8, 1, 1, 0];
        assert_eq!(ranking(&b, 2), BigUint::from(5u32));
    }

    #[test]
    fn example_14_graph_rank_7() {
        // edges (1,2), (1,3). vertices 1,2,3.
        let mut g = Graph::empty(3);
        g.rows[1] = vec![1]; // vertex 2 adjacent to 1
        g.rows[2] = vec![1, 0]; // vertex 3 adjacent to 1, not 2
        assert_eq!(graph_ranking(&g, 2), BigUint::from(7u32));
    }

    #[test]
    fn rank_unrank_strings_roundtrip() {
        for n in 1..=6 {
            for k in 0..=n {
                let list = pruned_brgc(n, k);
                assert_eq!(BigUint::from(list.len()), subset_count(n, k));
                for (idx, b) in list.iter().enumerate() {
                    let r = ranking(b, k);
                    assert_eq!(r, BigUint::from(idx + 1), "n={n} k={k} b={b:?}");
                    let back = unranking(n, k, &r);
                    assert_eq!(&back, b, "n={n} k={k} r={r}");
                }
            }
        }
    }

    #[test]
    fn consecutive_strings_are_gray() {
        for n in 1..=6 {
            for k in 0..=n {
                let list = pruned_brgc(n, k);
                let m = list.len();
                if m < 2 {
                    continue;
                }
                for i in 0..m {
                    let a = &list[i];
                    let b = &list[(i + 1) % m];
                    let diff: Vec<usize> = (0..n).filter(|&j| a[j] != b[j]).collect();
                    assert!(
                        diff.len() == 1 || diff.len() == 2,
                        "n={n} k={k} i={i} diff={diff:?}"
                    );
                    if diff.len() == 2 {
                        // a pivot flips one 1 to 0 and one 0 to 1
                        let ones_a: usize = a.iter().map(|x| *x as usize).sum();
                        let ones_b: usize = b.iter().map(|x| *x as usize).sum();
                        assert_eq!(ones_a, ones_b);
                    }
                }
            }
        }
    }

    #[test]
    fn graph_roundtrip_and_cyclic() {
        for n in 1usize..=4 {
            for k in 0..=n.saturating_sub(1) {
                let list = gen_wod(n, k);
                let m = lambda(n, k);
                assert_eq!(BigUint::from(list.len()), m, "n={n} k={k}");
                for (idx, g) in list.iter().enumerate() {
                    let r = graph_ranking(g, k);
                    assert_eq!(r, BigUint::from(idx + 1), "n={n} k={k} idx={idx}");
                    let back = graph_unranking(&r, n, k);
                    assert_eq!(&back, g);
                }
                if list.len() >= 2 {
                    for i in 0..list.len() {
                        let diff = edge_symdiff(&list[i], &list[(i + 1) % list.len()]);
                        assert!(
                            diff.len() == 1 || is_pivot(&diff),
                            "n={n} k={k} i={i} diff={diff:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lambda_even_when_required() {
        // Lemma 4: n ≥ 2 and k ≥ 1 implies λ even.
        for n in 2..=5 {
            for k in 1..=n - 1 {
                assert_eq!(&lambda(n, k) % 2u32, BigUint::zero());
            }
        }
        assert_eq!(lambda(1, 3), BigUint::one());
        assert_eq!(lambda(4, 0), BigUint::one());
    }
}
