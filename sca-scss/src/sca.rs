//! Public SCA solvers via the Theorem 1.1 reduction.

use crate::graph::DiGraph;
use crate::reduce::{build_unweighted, build_weighted};
use crate::scss::{scss_unweighted, scss_weighted};

fn contract_unweighted(
    g: &DiGraph,
    links: &[(usize, usize)],
) -> (DiGraph, Vec<(usize, usize)>) {
    if g.is_dag() {
        return (g.clone(), links.to_vec());
    }
    let (dag, id) = g.condensation();
    let mapped: Vec<(usize, usize)> = links
        .iter()
        .filter_map(|&(u, v)| {
            if u >= id.len() || v >= id.len() {
                return None;
            }
            let a = id[u];
            let b = id[v];
            if a != b {
                Some((a, b))
            } else {
                None
            }
        })
        .collect();
    (dag, mapped)
}

fn contract_weighted(
    g: &DiGraph,
    links: &[(usize, usize, u64)],
) -> (DiGraph, Vec<(usize, usize, u64)>) {
    if g.is_dag() {
        return (g.clone(), links.to_vec());
    }
    let (dag, id) = g.condensation();
    let mapped: Vec<(usize, usize, u64)> = links
        .iter()
        .filter_map(|&(u, v, w)| {
            if u >= id.len() || v >= id.len() {
                return None;
            }
            let a = id[u];
            let b = id[v];
            if a != b {
                Some((a, b, w))
            } else {
                None
            }
        })
        .collect();
    (dag, mapped)
}

/// Unweighted SCA: exists F subset of L, |F| <= k, such that G+F is strong.
pub fn sca_unweighted(g: &DiGraph, links: &[(usize, usize)], k: usize) -> bool {
    if g.is_strongly_connected() {
        return true;
    }
    let (dag, links) = contract_unweighted(g, links);
    if dag.is_strongly_connected() {
        return true;
    }
    let s = dag.sources();
    let t = dag.sinks();
    if s.len() > k || t.len() > k {
        return false;
    }
    let j = build_unweighted(&dag, &links);
    scss_unweighted(j.n, &j.edges, k as u64)
}

/// Weighted SCA: exists F subset of L with |F| <= k and w(F) <= t.
pub fn sca_weighted(g: &DiGraph, links: &[(usize, usize, u64)], k: usize, t: u64) -> bool {
    if g.is_strongly_connected() {
        return true;
    }
    let (dag, links) = contract_weighted(g, links);
    if dag.is_strongly_connected() {
        return true;
    }
    let s = dag.sources();
    let sinks = dag.sinks();
    if s.len() > k || sinks.len() > k {
        return false;
    }
    let j = build_weighted(&dag, &links, k);
    scss_weighted(j.n, &j.edges, k, t)
}

/// Terminal count |Z| after condensation, or 0 if already strong.
pub fn terminal_count(g: &DiGraph) -> usize {
    if g.is_strongly_connected() {
        return g.len().min(1);
    }
    let (dag, _) = g.condensation();
    dag.terminals().len()
}
