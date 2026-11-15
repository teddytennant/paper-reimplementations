//! Algorithm 1: exact greedy influence maximization via one forward+reverse pass
//! per seed (Theorem 4.1 / Theorem 5.1).

use crate::graph::Graph;
use crate::inference::influence_and_gradient;
use crate::treewidth::TreeDecomposition;

/// Exact greedy seed set of size `k`.
///
/// At each step set `θ ← 1_S`, evaluate `F(θ)` and `∇F`, and add
/// `v* = argmax_{v ∉ S} ∂F/∂θ_v`. Ties break toward the smallest index.
pub fn exact_greedy(graph: &Graph, td: &TreeDecomposition, k: usize) -> Vec<usize> {
    let n = graph.n();
    let k = k.min(n);
    let mut s = Vec::with_capacity(k);
    let mut in_s = vec![false; n];
    for _ in 0..k {
        let mut theta = vec![0.0; n];
        for &v in &s {
            theta[v] = 1.0;
        }
        let (_f, grad) = influence_and_gradient(graph, td, &theta);
        let mut best_v = None;
        let mut best_g = f64::NEG_INFINITY;
        for v in 0..n {
            if in_s[v] {
                continue;
            }
            if best_v.is_none() || grad[v] > best_g {
                best_g = grad[v];
                best_v = Some(v);
            }
        }
        let v = best_v.expect("no candidate");
        in_s[v] = true;
        s.push(v);
    }
    s
}
