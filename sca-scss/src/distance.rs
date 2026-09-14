//! Layered shortest paths (Lemma 2.1) and unweighted d_H.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::graph::DiGraph;

pub const INF: u64 = u64::MAX / 4;

pub fn add_cost(a: u64, b: u64) -> u64 {
    if a >= INF || b >= INF {
        INF
    } else {
        a.saturating_add(b).min(INF)
    }
}

fn link_adj_unweighted(n: usize, links: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in links {
        if u < n && v < n {
            adj[u].push(v);
        }
    }
    adj
}

fn link_adj_weighted(n: usize, links: &[(usize, usize, u64)]) -> Vec<Vec<(usize, u64)>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v, w) in links {
        if u < n && v < n {
            adj[u].push((v, w));
        }
    }
    adj
}

/// 0-1 BFS: base edges cost 0, links cost 1.
/// Returns (dist, parent) where parent[v] = Some((prev, via_link)).
pub fn unweighted_shortest(
    g: &DiGraph,
    links: &[(usize, usize)],
    src: usize,
) -> (Vec<u64>, Vec<Option<(usize, bool)>>) {
    let n = g.len();
    let mut dist = vec![INF; n];
    let mut parent = vec![None; n];
    if src >= n {
        return (dist, parent);
    }
    let link_adj = link_adj_unweighted(n, links);
    dist[src] = 0;
    let mut dq = VecDeque::new();
    dq.push_back(src);
    while let Some(v) = dq.pop_front() {
        let dv = dist[v];
        for &w in g.out(v) {
            if dv < dist[w] {
                dist[w] = dv;
                parent[w] = Some((v, false));
                dq.push_front(w);
            }
        }
        for &w in &link_adj[v] {
            let nd = add_cost(dv, 1);
            if nd < dist[w] {
                dist[w] = nd;
                parent[w] = Some((v, true));
                dq.push_back(w);
            }
        }
    }
    (dist, parent)
}

pub fn d_h_unweighted(g: &DiGraph, links: &[(usize, usize)], u: usize, v: usize) -> Option<u64> {
    let (dist, _) = unweighted_shortest(g, links, u);
    if dist[v] >= INF {
        None
    } else {
        Some(dist[v])
    }
}

/// Reconstruct a min-link u-to-v path as a list of hops (from, to, via_link).
pub fn min_link_path(
    g: &DiGraph,
    links: &[(usize, usize)],
    src: usize,
    dst: usize,
) -> Option<Vec<(usize, usize, bool)>> {
    if src == dst {
        return Some(Vec::new());
    }
    let (dist, parent) = unweighted_shortest(g, links, src);
    if dist[dst] >= INF {
        return None;
    }
    let mut hops = Vec::new();
    let mut cur = dst;
    while cur != src {
        let (prev, via_link) = parent[cur]?;
        hops.push((prev, cur, via_link));
        cur = prev;
    }
    hops.reverse();
    Some(hops)
}

/// Lemma 2.1: `out[h][v]` is the min total link weight of a path from `src`
/// to `v` that uses at most `h` links, or INF if none exists.
pub fn layered_at_most(
    g: &DiGraph,
    links: &[(usize, usize, u64)],
    src: usize,
    k: usize,
) -> Vec<Vec<u64>> {
    let n = g.len();
    let mut exact = vec![vec![INF; n]; k + 1];
    if src >= n {
        return exact;
    }
    let link_adj = link_adj_weighted(n, links);
    exact[0][src] = 0;
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0u64, 0usize, src)));
    while let Some(Reverse((d, h, v))) = heap.pop() {
        if d != exact[h][v] {
            continue;
        }
        for &w in g.out(v) {
            if d < exact[h][w] {
                exact[h][w] = d;
                heap.push(Reverse((d, h, w)));
            }
        }
        if h < k {
            for &(to, wt) in &link_adj[v] {
                let nd = add_cost(d, wt);
                if nd < exact[h + 1][to] {
                    exact[h + 1][to] = nd;
                    heap.push(Reverse((nd, h + 1, to)));
                }
            }
        }
    }
    let mut at_most = exact;
    for h in 1..=k {
        for v in 0..n {
            if at_most[h - 1][v] < at_most[h][v] {
                at_most[h][v] = at_most[h - 1][v];
            }
        }
    }
    at_most
}

pub fn d_h(
    g: &DiGraph,
    links: &[(usize, usize, u64)],
    u: usize,
    v: usize,
    h: usize,
) -> Option<u64> {
    let table = layered_at_most(g, links, u, h);
    let w = table[h][v];
    if w >= INF {
        None
    } else {
        Some(w)
    }
}
