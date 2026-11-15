//! Lemma 4.1 and Lemma 4.2 of arXiv:2608.30299, checked on the output of
//! [`dcluster`] over a sweep of graph families, sizes and seeds.
//!
//! Which properties are asserted outright and which are asserted as a rate:
//!
//! * (2), (3), `F` contained in `E(G)`, and "each cluster is a connected
//!   component of `(V, F)` holding exactly one center" hold on every run in the
//!   sweep and are asserted directly. The paper states all four properties with
//!   probability `1 - O(log n / n^2)`, but the argument for (2) and (3) is
//!   structural: line 16 removes the whole of `X_i` in iteration `i`, so
//!   `V(G_i)` never contains a vertex of `X_{i+1}`, which pins (2) down without
//!   reference to the sampling, and (3) follows from the depth bound in the
//!   proof.
//! * (1) and (4) and the Lemma 4.2 message bound genuinely depend on the coin
//!   flips. (1) fails when a vertex of `X_i` finds neither a neighboring center
//!   nor a removed neighbor among its probes, and line 16 removes it anyway.
//!   (4) fails when `|W_i|` overshoots twice its mean. Those are asserted as a
//!   failure rate with room to spare, not as "never".
//!
//! Every case is seeded, so a case that passes once passes every time.

use congest_dcluster::dcluster::{
    ceil_log2, cluster_count_bound, dcluster, delta, diameter_bound, i_max, message_bound, Clustering,
};
use congest_dcluster::graph::{self, Graph};
use congest_dcluster::partition::induced_diameter;
use std::collections::BTreeSet;

/// The three probabilistic outcomes of one run.
#[derive(Default, Debug)]
struct Outcome {
    unclustered_vertices: bool,
    class_count_over_bound: bool,
    messages_over_bound: bool,
}

fn families(n: usize) -> Vec<(String, Graph)> {
    let mut out = vec![
        ("path".to_string(), graph::path(n)),
        ("cycle".to_string(), graph::cycle(n)),
        ("star".to_string(), graph::star(n)),
        ("clique".to_string(), graph::clique(n)),
        ("barbell".to_string(), graph::barbell(n)),
    ];
    for (tag, p) in [("er0.05", 0.05), ("er0.2", 0.2), ("er0.5", 0.5), ("er0.8", 0.8)] {
        out.push((tag.to_string(), graph::erdos_renyi(n, p, 0x51e5 ^ n as u64)));
    }
    out.push(("pa3".to_string(), graph::preferential_attachment(n, 3, 0xbeef ^ n as u64)));
    out
}

/// Adjacency lists of `(V, F)`.
fn f_adjacency(n: usize, f: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v) in f {
        adj[u as usize].push(v);
        adj[v as usize].push(u);
    }
    adj
}

/// Connected components of `(V, F)`, as a component id per vertex.
fn components(n: usize, adj: &[Vec<u32>]) -> (Vec<usize>, usize) {
    let mut comp = vec![usize::MAX; n];
    let mut next = 0;
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let mut stack = vec![s];
        comp[s] = next;
        while let Some(u) = stack.pop() {
            for &w in &adj[u] {
                if comp[w as usize] == usize::MAX {
                    comp[w as usize] = next;
                    stack.push(w as usize);
                }
            }
        }
        next += 1;
    }
    (comp, next)
}

/// Largest cluster diameter in `(V, F)` over a whole run, 0 if every cluster is
/// a singleton.
fn observed_diameter(g: &Graph, c: &Clustering) -> u32 {
    if !c.is_total() {
        return 0;
    }
    let adj = f_adjacency(g.n(), &c.f);
    let part_of: Vec<usize> = (0..g.n()).map(|v| c.cluster_of[v].unwrap()).collect();
    c.clusters
        .iter()
        .enumerate()
        .filter_map(|(idx, cl)| induced_diameter(&adj, &cl.members, &part_of, idx))
        .max()
        .unwrap_or(0)
}

/// Check one run and report the probabilistic outcomes. Everything that should
/// hold on every run is asserted here.
fn check(label: &str, g: &Graph, seed: u64, c: &Clustering) -> Outcome {
    let n = g.n();
    let ctx = format!("{label} n={n} seed={seed}");
    let mut outcome = Outcome::default();

    // F is a subset of E(G), with no repeats and no self loops.
    let edges: BTreeSet<(u32, u32)> = g.edges().into_iter().collect();
    let f_set: BTreeSet<(u32, u32)> = c.f.iter().copied().collect();
    assert_eq!(f_set.len(), c.f.len(), "{ctx}: F has a repeated edge");
    for &(u, v) in &c.f {
        assert!(u < v, "{ctx}: F edge {u},{v} is not normalized");
        assert!(edges.contains(&(u, v)), "{ctx}: F edge {u},{v} is not in E(G)");
    }

    let adj = f_adjacency(n, &c.f);
    let (comp, comp_count) = components(n, &adj);

    // Lemma 4.1(1), first half: the collections C^(i) are mutually disjoint,
    // that is, every cluster shows up in exactly one class, and no vertex is a
    // member of two clusters.
    let mut seen_cluster = vec![0usize; c.cluster_count()];
    for (i, class) in c.classes.iter().enumerate() {
        for &idx in class {
            seen_cluster[idx] += 1;
            assert_eq!(c.clusters[idx].class, i, "{ctx}: cluster {idx} filed under two classes");
        }
    }
    assert!(
        seen_cluster.iter().all(|&k| k == 1),
        "{ctx}: some cluster is in zero or two of the collections C^(i)"
    );
    let mut membership = vec![0usize; n];
    for cluster in &c.clusters {
        for &v in &cluster.members {
            membership[v as usize] += 1;
        }
    }
    assert!(
        membership.iter().all(|&k| k <= 1),
        "{ctx}: a vertex is a member of two clusters"
    );

    // Lemma 4.1(1), second half: every vertex lands in a cluster. This is the
    // part that can fail, so it is recorded rather than asserted.
    outcome.unclustered_vertices = !c.is_total();
    if outcome.unclustered_vertices {
        return outcome;
    }

    // Each cluster is exactly one connected component of (V, F), and it holds
    // exactly one center.
    assert_eq!(
        comp_count,
        c.cluster_count(),
        "{ctx}: (V, F) has {comp_count} components but there are {} clusters",
        c.cluster_count()
    );
    let mut centers_per_component = vec![0usize; comp_count];
    for cluster in &c.clusters {
        centers_per_component[comp[cluster.center as usize]] += 1;
        let home = comp[cluster.center as usize];
        for &v in &cluster.members {
            assert_eq!(
                comp[v as usize], home,
                "{ctx}: vertex {v} is in cluster {} but a different component of (V, F)",
                cluster.center
            );
        }
    }
    assert!(
        centers_per_component.iter().all(|&k| k == 1),
        "{ctx}: some component of (V, F) has zero or two centers"
    );

    // Lemma 4.1(2): deg_G(v) <= 2 Delta_i for every v in a cluster of C^(i).
    for cluster in &c.clusters {
        let limit = 2 * delta(cluster.class);
        for &v in &cluster.members {
            assert!(
                g.degree(v as usize) as u64 <= limit,
                "{ctx}: vertex {v} has degree {} in a class-{} cluster, over 2 Delta_i = {limit}",
                g.degree(v as usize),
                cluster.class
            );
        }
    }

    // Lemma 4.1(3): the subgraph of (V, F) induced by each cluster has diameter
    // at most 2 ceil(log2 n) - 2, measured by breadth-first search inside the
    // cluster rather than assumed.
    let part_of: Vec<usize> = (0..n).map(|v| c.cluster_of[v].unwrap()).collect();
    let bound = diameter_bound(n);
    for (idx, cluster) in c.clusters.iter().enumerate() {
        let d = induced_diameter(&adj, &cluster.members, &part_of, idx)
            .unwrap_or_else(|| panic!("{ctx}: cluster {idx} is disconnected in (V, F)"));
        assert!(
            d <= bound,
            "{ctx}: cluster {idx} has diameter {d} in (V, F), over 2 ceil(log2 n) - 2 = {bound}"
        );
    }

    // Lemma 4.1(4): |C^(i)| <= 12 n ln n / Delta_i.
    for i in 0..=i_max(n) {
        if c.classes[i].len() as f64 > cluster_count_bound(n, i) {
            outcome.class_count_over_bound = true;
        }
    }

    // Lemma 4.2: at most 48 n ln n ceil(log2 n) messages.
    outcome.messages_over_bound = c.messages as f64 > message_bound(n);

    // The round structure of Section 4.1 does not depend on the input.
    assert_eq!(c.rounds, 4 * ceil_log2(n) as u64, "{ctx}: unexpected round count");

    outcome
}

#[test]
fn lemma_41_and_42_over_a_sweep_of_graphs() {
    let mut runs = 0usize;
    let mut unclustered = 0usize;
    let mut class_over = 0usize;
    let mut messages_over = 0usize;
    let mut worst_message_ratio = 0.0f64;
    let mut nontrivial = 0usize;
    let mut largest_cluster = 0usize;
    let mut deepest = 0u32;

    for &n in &[16usize, 33, 64, 100, 128] {
        for (label, g) in families(n) {
            for seed in 0..8u64 {
                let c = dcluster(&g, seed);
                let out = check(&label, &g, seed, &c);
                runs += 1;
                unclustered += usize::from(out.unclustered_vertices);
                class_over += usize::from(out.class_count_over_bound);
                messages_over += usize::from(out.messages_over_bound);
                worst_message_ratio = worst_message_ratio.max(c.messages as f64 / message_bound(n));
                let biggest = c.clusters.iter().map(|cl| cl.members.len()).max().unwrap_or(0);
                largest_cluster = largest_cluster.max(biggest);
                if biggest >= 3 {
                    nontrivial += 1;
                }
                deepest = deepest.max(observed_diameter(&g, &c));
            }
        }
    }

    println!("{runs} runs, worst message ratio against the Lemma 4.2 bound {worst_message_ratio:.4}");
    println!("failures: {unclustered} unclustered, {class_over} class-count, {messages_over} message-count");
    println!("{nontrivial} runs built a cluster of 3 or more vertices, largest {largest_cluster}, deepest diameter {deepest}");

    // Guard against a vacuous sweep. On the sparse families every vertex ends
    // up as the center of its own singleton cluster, which is what Algorithm 1
    // does when 6 ln n / Delta_i saturates at 1, and those runs never exercise
    // an edge of F. The dense families have to carry the diameter and
    // connectivity checks, so insist that enough of them did.
    //
    // The deepest cluster seen here has diameter 2, well inside the
    // 2 ceil(log2 n) - 2 of Lemma 4.1(3). A deeper one needs a probe on line 11
    // to land on a vertex that itself joined through a probe, and that chain
    // does not show up at these sizes.
    assert!(nontrivial > runs / 8, "the sweep barely built any real clusters");
    assert!(largest_cluster >= 8);
    assert!(deepest >= 2);

    // Lemma 4.1 and Lemma 4.2 both hold with probability 1 - O(log n / n^2),
    // so a failure is allowed. At n >= 16 over this many runs it should be far
    // under one percent, and in practice the sweep comes out clean.
    let rate = |k: usize| k as f64 / runs as f64;
    assert!(rate(unclustered) < 0.01, "{unclustered} of {runs} runs left a vertex unclustered");
    assert!(rate(class_over) < 0.01, "{class_over} of {runs} runs broke Lemma 4.1(4)");
    assert!(rate(messages_over) < 0.01, "{messages_over} of {runs} runs broke Lemma 4.2");
    assert!(worst_message_ratio < 1.0, "some run exceeded the Lemma 4.2 bound");
}

#[test]
fn lemma_41_and_42_over_many_seeds_on_one_graph() {
    // The sweep above varies the graph. This one holds the graph still and
    // varies only the randomness, which is what the "with high probability" in
    // Lemma 4.1 quantifies over.
    let n = 96;
    let g = graph::erdos_renyi(n, 0.35, 2024);
    let mut unclustered = 0usize;
    let mut class_over = 0usize;
    let mut messages_over = 0usize;
    let seeds = 400u64;

    for seed in 0..seeds {
        let c = dcluster(&g, seed);
        let out = check("er0.35", &g, seed, &c);
        unclustered += usize::from(out.unclustered_vertices);
        class_over += usize::from(out.class_count_over_bound);
        messages_over += usize::from(out.messages_over_bound);
    }

    println!(
        "{seeds} seeds: {unclustered} unclustered, {class_over} class-count, {messages_over} message-count"
    );
    let rate = |k: usize| k as f64 / seeds as f64;
    assert!(rate(unclustered) < 0.01);
    assert!(rate(class_over) < 0.01);
    assert_eq!(messages_over, 0);
}

#[test]
fn message_count_sits_well_under_the_lemma_42_bound() {
    // The dense cases are where the announcements on lines 7 to 9 cost the
    // most: |W_i| * 2 Delta_i per iteration in the proof of Lemma 4.2.
    for &n in &[32usize, 64, 128, 200] {
        for family in [graph::clique(n), graph::barbell(n), graph::erdos_renyi(n, 0.6, 77)] {
            let c = dcluster(&family, 5);
            let bound = message_bound(n);
            assert!(
                (c.messages as f64) <= bound,
                "n = {n}: {} messages against the bound {bound:.0}",
                c.messages
            );
        }
    }
}

#[test]
fn a_vertex_only_learns_through_its_own_edges() {
    // The network refuses a send along a non-edge, so the only way the sweep
    // above can pass is by messages travelling on edges of G. This pins the
    // claim down on the F side: every edge DCluster puts in F is incident to
    // the vertex that added it.
    let g = graph::preferential_attachment(128, 4, 3);
    let c = dcluster(&g, 19);
    for v in 0..g.n() {
        if let Some(parent) = c.parent[v] {
            assert!(
                g.is_neighbor(v, parent as usize),
                "vertex {v} attached itself to a non-neighbor"
            );
        }
    }
}

#[test]
fn small_graphs_are_handled() {
    // At these sizes 6 ln n is larger than every Delta_i, so line 5 samples
    // the whole of V(G_i) in the first iteration and every vertex becomes the
    // center of its own singleton cluster. Nothing is left to fail.
    for n in 2..=12usize {
        for seed in 0..20u64 {
            for (label, g) in [
                ("path", graph::path(n)),
                ("cycle", graph::cycle(n)),
                ("star", graph::star(n)),
                ("clique", graph::clique(n)),
            ] {
                let c = dcluster(&g, seed);
                let out = check(label, &g, seed, &c);
                assert!(!out.unclustered_vertices, "{label} n = {n} seed = {seed}");
                assert!(!out.class_count_over_bound, "{label} n = {n} seed = {seed}");
                assert!(!out.messages_over_bound, "{label} n = {n} seed = {seed}");
            }
        }
    }
}
