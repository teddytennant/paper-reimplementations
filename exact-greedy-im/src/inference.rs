//! Exact IC inference by separator-relation DP, two-sweep, and reverse-mode AD.
//!
//! Implements Lemmas 3.2–3.4, the variable source edges of §4.1, and Theorem 4.1.

use crate::autodiff::Tape;
use crate::graph::Graph;
use crate::relations::{self, MAX_AUG_BAG};
use crate::treewidth::TreeDecomposition;
use std::collections::HashMap;

/// Distribution over relations: relation-id → tape node of probability mass.
type Dist = HashMap<u64, usize>;

fn accum(tape: &mut Tape, dist: &mut Dist, rel: u64, node: usize) {
    if let Some(&old) = dist.get(&rel) {
        dist.insert(rel, tape.add(old, node));
    } else {
        dist.insert(rel, node);
    }
}

fn aug_bag(bag: &[usize], rho: usize) -> Vec<usize> {
    let mut v = bag.to_vec();
    v.push(rho);
    v.sort_unstable();
    v
}

fn separator(bx: &[usize], by: &[usize], rho: usize) -> Vec<usize> {
    let mut s: Vec<usize> = bx.iter().copied().filter(|v| by.contains(v)).collect();
    s.push(rho);
    s.sort_unstable();
    s
}

/// Combine a bag distribution with an incoming separator message (sequential Φ).
fn combine(tape: &mut Tape, d: &Dist, msg: &Dist, bag: &[usize], sep: &[usize]) -> Dist {
    let k = bag.len();
    let mut out = Dist::new();
    for (&r_d, &n_d) in d {
        for (&r_m, &n_m) in msg {
            let lifted = relations::embed(r_m, sep, bag);
            let tc = relations::union_then_tc(r_d, lifted, k);
            let prod = tape.mul(n_d, n_m);
            accum(tape, &mut out, tc, prod);
        }
    }
    out
}

fn restrict_dist(tape: &mut Tape, d: &Dist, from: &[usize], to: &[usize]) -> Dist {
    let mut out = Dist::new();
    for (&rel, &node) in d {
        let r2 = relations::restrict(rel, from, to);
        accum(tape, &mut out, r2, node);
    }
    out
}

/// Local bag distribution `P^loc_x` over relations on `B_x ∪ {ρ}`.
///
/// Original owned edges are enumerated exactly (`2^{#owned}`). Each artificial
/// source edge `ρ → v` then splits every mass `p` into `p(1-θ_v)` and `p θ_v`.
/// **Both branches are kept even when `θ_v ∈ {0, 1}`** (paper §4.3).
fn local_distribution(
    tape: &mut Tape,
    bag: &[usize],
    owned_orig: &[(usize, usize, f64)],
    owned_sources: &[usize],
    theta: &[usize],
    one_minus_theta: &[usize],
    rho: usize,
) -> Dist {
    let k_orig = owned_orig.len();
    let mut orig: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for mask in 0..(1u32 << k_orig) {
        let mut mass = tape.constant(1.0);
        let mut edges = Vec::new();
        for i in 0..k_orig {
            let (u, v, p) = owned_orig[i];
            if mask & (1 << i) != 0 {
                let c = tape.constant(p);
                mass = tape.mul(mass, c);
                edges.push((u, v));
            } else {
                let c = tape.constant(1.0 - p);
                mass = tape.mul(mass, c);
            }
        }
        orig.push((mass, edges));
    }

    let k_src = owned_sources.len();
    let mut dist = Dist::new();
    for (mass, edges) in orig {
        for smask in 0..(1u32 << k_src) {
            let mut m = mass;
            let mut e = edges.clone();
            for i in 0..k_src {
                let v = owned_sources[i];
                if smask & (1 << i) != 0 {
                    m = tape.mul(m, theta[v]);
                    e.push((rho, v));
                } else {
                    m = tape.mul(m, one_minus_theta[v]);
                }
            }
            let rel = relations::from_edges(bag, &e);
            accum(tape, &mut dist, rel, m);
        }
    }
    dist
}

fn assign_edge_owners(graph: &Graph, td: &TreeDecomposition) -> Vec<usize> {
    graph
        .edges()
        .iter()
        .map(|e| {
            td.bags
                .iter()
                .position(|b| b.contains(&e.from) && b.contains(&e.to))
                .unwrap_or_else(|| {
                    panic!(
                        "edge {}->{} is not covered by the decomposition",
                        e.from, e.to
                    )
                })
        })
        .collect()
}

fn assign_source_owners(n: usize, td: &TreeDecomposition) -> Vec<usize> {
    (0..n)
        .map(|v| {
            td.bags
                .iter()
                .position(|b| b.contains(&v))
                .unwrap_or_else(|| panic!("vertex {v} is not covered by the decomposition"))
        })
        .collect()
}

fn dfs_postorder(adj: &[Vec<usize>]) -> (Vec<usize>, Vec<Option<usize>>) {
    let m = adj.len();
    let mut parent = vec![None; m];
    let mut post = Vec::new();
    let mut seen = vec![false; m];

    fn rec(
        u: usize,
        adj: &[Vec<usize>],
        parent: &mut [Option<usize>],
        seen: &mut [bool],
        post: &mut Vec<usize>,
    ) {
        seen[u] = true;
        for &v in &adj[u] {
            if !seen[v] {
                parent[v] = Some(u);
                rec(v, adj, parent, seen, post);
            }
        }
        post.push(u);
    }

    if m > 0 {
        rec(0, adj, &mut parent, &mut seen, &mut post);
        for u in 0..m {
            if !seen[u] {
                rec(u, adj, &mut parent, &mut seen, &mut post);
            }
        }
    }
    (post, parent)
}

struct Prepared {
    rho: usize,
    aug: Vec<Vec<usize>>,
    p_loc: Vec<Dist>,
    adj: Vec<Vec<usize>>,
}

fn prepare(graph: &Graph, td: &TreeDecomposition, theta: &[f64], tape: &mut Tape) -> Prepared {
    assert_eq!(theta.len(), graph.n());
    if let Err(e) = td.validate(graph) {
        panic!("invalid tree decomposition: {e}");
    }
    assert!(
        td.bags.iter().all(|b| b.len() <= 3),
        "only width ≤ 2 is implemented"
    );

    let n = graph.n();
    let rho = n;
    let m = td.bags.len();
    let edge_owner = assign_edge_owners(graph, td);
    let source_owner = assign_source_owners(n, td);

    let mut theta_nodes = Vec::with_capacity(n);
    let mut one_minus = Vec::with_capacity(n);
    for v in 0..n {
        let t = tape.param(v, theta[v]);
        theta_nodes.push(t);
        one_minus.push(tape.one_minus(t));
    }

    let aug: Vec<Vec<usize>> = td.bags.iter().map(|b| aug_bag(b, rho)).collect();
    for bag in &aug {
        assert!(bag.len() <= MAX_AUG_BAG);
    }

    let mut p_loc = Vec::with_capacity(m);
    for x in 0..m {
        let owned_orig: Vec<(usize, usize, f64)> = graph
            .edges()
            .iter()
            .enumerate()
            .filter(|(i, _)| edge_owner[*i] == x)
            .map(|(_, e)| (e.from, e.to, e.p))
            .collect();
        let owned_sources: Vec<usize> = (0..n).filter(|&v| source_owner[v] == x).collect();
        p_loc.push(local_distribution(
            tape,
            &aug[x],
            &owned_orig,
            &owned_sources,
            &theta_nodes,
            &one_minus,
            rho,
        ));
    }

    Prepared {
        rho,
        aug,
        p_loc,
        adj: td.bag_adj(),
    }
}

fn compute_message(
    tape: &mut Tape,
    x: usize,
    y: usize,
    td: &TreeDecomposition,
    prep: &Prepared,
    messages: &HashMap<(usize, usize), Dist>,
) -> Dist {
    let bag = &prep.aug[x];
    let mut d = prep.p_loc[x].clone();
    for &z in &prep.adj[x] {
        if z == y {
            continue;
        }
        let sep = separator(&td.bags[x], &td.bags[z], prep.rho);
        let msg = messages.get(&(z, x)).expect("missing incoming message");
        d = combine(tape, &d, msg, bag, &sep);
    }
    let cxy = separator(&td.bags[x], &td.bags[y], prep.rho);
    restrict_dist(tape, &d, bag, &cxy)
}

fn two_sweep(
    tape: &mut Tape,
    td: &TreeDecomposition,
    prep: &Prepared,
) -> HashMap<(usize, usize), Dist> {
    let (post, parent) = dfs_postorder(&prep.adj);
    let mut messages: HashMap<(usize, usize), Dist> = HashMap::new();

    // Upward: leaves toward the root (Lemma 3.3).
    for &x in &post {
        if let Some(y) = parent[x] {
            let msg = compute_message(tape, x, y, td, prep, &messages);
            messages.insert((x, y), msg);
        }
    }

    // Downward: root toward the leaves.
    for &x in post.iter().rev() {
        for &z in &prep.adj[x] {
            if parent.get(z).copied().flatten() == Some(x) {
                let msg = compute_message(tape, x, z, td, prep, &messages);
                messages.insert((x, z), msg);
            }
        }
    }
    messages
}

fn global_bag_dist(
    tape: &mut Tape,
    td: &TreeDecomposition,
    prep: &Prepared,
    x: usize,
    messages: &HashMap<(usize, usize), Dist>,
) -> Dist {
    let bag = &prep.aug[x];
    let mut d = prep.p_loc[x].clone();
    for &z in &prep.adj[x] {
        let sep = separator(&td.bags[x], &td.bags[z], prep.rho);
        let msg = messages.get(&(z, x)).expect("missing incoming message");
        d = combine(tape, &d, msg, bag, &sep);
    }
    d
}

fn rho_reaches(rel: u64, bag: &[usize], v: usize, rho: usize) -> bool {
    let k = bag.len();
    let i = relations::index_of(bag, rho);
    let j = relations::index_of(bag, v);
    relations::reaches(rel, i, j, k)
}

/// `F(θ) = ∑_u P_θ(ρ ⇝ u)` and `∇F` by reverse-mode AD through the DP circuit.
pub fn influence_and_gradient(
    graph: &Graph,
    td: &TreeDecomposition,
    theta: &[f64],
) -> (f64, Vec<f64>) {
    let n = graph.n();
    if n == 0 {
        return (0.0, vec![]);
    }
    let mut tape = Tape::new(n);
    let prep = prepare(graph, td, theta, &mut tape);
    let messages = two_sweep(&mut tape, td, &prep);

    let m = td.bags.len();
    let mut globals = Vec::with_capacity(m);
    for x in 0..m {
        globals.push(global_bag_dist(&mut tape, td, &prep, x, &messages));
    }

    let mut claimed = vec![false; n];
    let mut f = tape.constant(0.0);
    for x in 0..m {
        for &v in &td.bags[x] {
            if claimed[v] {
                continue;
            }
            claimed[v] = true;
            let mut pi = tape.constant(0.0);
            for (&rel, &node) in &globals[x] {
                if rho_reaches(rel, &prep.aug[x], v, prep.rho) {
                    pi = tape.add(pi, node);
                }
            }
            f = tape.add(f, pi);
        }
    }
    debug_assert!(claimed.iter().all(|&c| c));

    let value = tape.value(f);
    let grad = tape.reverse(f);
    (value, grad)
}

/// Influence spread `σ(S) = F(1_S)`.
pub fn influence(graph: &Graph, td: &TreeDecomposition, seeds: &[usize]) -> f64 {
    let n = graph.n();
    let mut theta = vec![0.0; n];
    for &s in seeds {
        assert!(s < n);
        theta[s] = 1.0;
    }
    influence_and_gradient(graph, td, &theta).0
}
