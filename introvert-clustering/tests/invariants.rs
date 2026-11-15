//! Invariant tests against Chang and Dolatabadi, arXiv:2609.10044.

use introvert_clustering::{
    introvert_cluster, layered_decompose, mpx_cluster, trim, trim_with_trace, validate_beta,
    validate_eps, validate_lambda, Clustering, Graph,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const EPS: f64 = 0.1;
const BETA: f64 = 0.5;

fn path(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n.saturating_sub(1) {
        g.add_edge(i, i + 1).expect("path edge");
    }
    g
}

fn cycle(n: usize) -> Graph {
    let mut g = path(n);
    if n >= 3 {
        g.add_edge(0, n - 1).expect("cycle edge");
    }
    g
}

fn complete(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for u in 0..n {
        for v in (u + 1)..n {
            g.add_edge(u, v).expect("complete edge");
        }
    }
    g
}

fn star(n: usize) -> Graph {
    let mut g = Graph::new(n);
    if n == 0 {
        return g;
    }
    for v in 1..n {
        g.add_edge(0, v).expect("star edge");
    }
    g
}

fn grid(rows: usize, cols: usize) -> Graph {
    let mut g = Graph::new(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                g.add_edge(v, v + 1).expect("grid horiz");
            }
            if r + 1 < rows {
                g.add_edge(v, v + cols).expect("grid vert");
            }
        }
    }
    g
}

fn disjoint_cliques(sizes: &[usize]) -> Graph {
    let n: usize = sizes.iter().sum();
    let mut g = Graph::new(n);
    let mut offset = 0;
    for &s in sizes {
        for u in 0..s {
            for v in (u + 1)..s {
                g.add_edge(offset + u, offset + v).expect("clique edge");
            }
        }
        offset += s;
    }
    g
}

fn gnp(n: usize, p: f64, rng: &mut impl Rng) -> Graph {
    let mut g = Graph::new(n);
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                g.add_edge(u, v).expect("gnp edge");
            }
        }
    }
    g
}

fn all_one_cluster(n: usize) -> Clustering {
    Clustering::from_assignment(vec![Some(0); n])
}

fn sample_graphs(rng: &mut impl Rng) -> Vec<Graph> {
    vec![
        path(7),
        cycle(8),
        complete(6),
        grid(3, 4),
        gnp(12, 0.35, rng),
        star(10),
        disjoint_cliques(&[3, 4, 5]),
        Graph::new(0),
        Graph::new(1),
        Graph::from_edges(4, &[]).expect("edgeless"),
    ]
}

fn rng_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Lemma 1: after trim, every remaining clustered v satisfies the introvert property.
#[test]
fn lemma1_introvert_property() {
    let mut rng = rng_seed(7);
    for g in sample_graphs(&mut rng) {
        let c0 = all_one_cluster(g.n());
        let c1 = trim(&g, &c0, EPS).expect("trim");
        assert!(
            c1.introvert_ok(&g, EPS),
            "introvert property failed after trim, n={}",
            g.n()
        );

        if g.n() >= 2 && g.m() > 0 {
            let mpx = mpx_cluster(&g, 0.2, &mut rng).expect("mpx");
            let trimmed = trim(&g, &mpx, EPS).expect("trim mpx");
            assert!(trimmed.introvert_ok(&g, EPS));
        }
    }
}

/// Lemma 1: U1 <= U0 / (2 eps). Also a hand-built clustering that trim actually changes.
#[test]
fn lemma1_cut_bound() {
    let mut rng = rng_seed(11);
    for g in sample_graphs(&mut rng) {
        let c0 = all_one_cluster(g.n());
        let u0 = c0.inter_cluster_edge_count(&g);
        let c1 = trim(&g, &c0, EPS).expect("trim");
        let u1 = c1.inter_cluster_edge_count(&g);
        assert!(
            (u1 as f64) <= (u0 as f64) / (2.0 * EPS) + 1e-9,
            "cut bound failed: U1={u1} U0={u0}"
        );

        if g.m() > 0 {
            let mpx = mpx_cluster(&g, 0.25, &mut rng).expect("mpx");
            let u0 = mpx.inter_cluster_edge_count(&g);
            let c1 = trim(&g, &mpx, EPS).expect("trim");
            let u1 = c1.inter_cluster_edge_count(&g);
            assert!((u1 as f64) <= (u0 as f64) / (2.0 * EPS) + 1e-9);
        }
    }

    // K_5 with two clusters of sizes 2 and 3: din=1, deg=4 violates introvert.
    let g = complete(5);
    let mut assign = vec![Some(0); 5];
    assign[3] = Some(1);
    assign[4] = Some(1);
    let c0 = Clustering::from_assignment(assign);
    assert!(!c0.introvert_ok(&g, EPS));
    let u0 = c0.inter_cluster_edge_count(&g);
    let trace = trim_with_trace(&g, &c0, EPS).expect("trim");
    assert!(!trace.removed.is_empty(), "trim should remove violators");
    let u1 = trace.clustering.inter_cluster_edge_count(&g);
    assert!((u1 as f64) <= (u0 as f64) / (2.0 * EPS) + 1e-9);
    assert!(trace.clustering.introvert_ok(&g, EPS));
}

/// Equation (4): at removal, din < (1/2 - eps) deg, hence
/// dout > ((1+2eps)/(1-2eps)) din.
#[test]
fn equation4_on_removed_vertices() {
    let mut rng = rng_seed(19);
    let graphs = sample_graphs(&mut rng);
    let mut saw_removal = false;
    for g in graphs {
        let mpx = if g.n() == 0 {
            Clustering::new(0)
        } else {
            mpx_cluster(&g, 0.3, &mut rng).expect("mpx")
        };
        let trace = trim_with_trace(&g, &mpx, EPS).expect("trim");
        for rem in &trace.removed {
            saw_removal |= rem.deg > 0;
            let din = rem.din as f64;
            let deg = rem.deg as f64;
            let dout = deg - din;
            assert!(
                din < (0.5 - EPS) * deg,
                "v={} din={din} deg={deg} should violate introvert at removal",
                rem.vertex
            );
            let ratio = (1.0 + 2.0 * EPS) / (1.0 - 2.0 * EPS);
            assert!(
                dout > ratio * din - 1e-12,
                "equation (4) failed for v={}: dout={dout} din={din}",
                rem.vertex
            );
        }
    }

    let g = complete(5);
    let mut assign = vec![Some(0); 5];
    assign[3] = Some(1);
    assign[4] = Some(1);
    let trace = trim_with_trace(&g, &Clustering::from_assignment(assign), EPS).expect("trim");
    assert!(!trace.removed.is_empty());
    for rem in &trace.removed {
        let din = rem.din as f64;
        let deg = rem.deg as f64;
        let dout = deg - din;
        assert!(din < (0.5 - EPS) * deg);
        let ratio = (1.0 + 2.0 * EPS) / (1.0 - 2.0 * EPS);
        assert!(dout > ratio * din - 1e-12);
        saw_removal = true;
    }
    assert!(saw_removal);
}

/// Definition 1 local property after Lemma 2.
#[test]
fn lemma2_introvert_ok() {
    let mut rng = rng_seed(23);
    for g in sample_graphs(&mut rng) {
        let c = introvert_cluster(&g, EPS, BETA, &mut rng).expect("lemma 2");
        assert!(c.introvert_ok(&g, EPS));
        for cluster in c.clusters() {
            assert!(!cluster.is_empty());
        }
    }
}

/// Theorem 7: MPX clusters are pairwise disjoint, nonempty, and G[C]-connected.
#[test]
fn theorem7_diameter_and_partition() {
    let mut rng = rng_seed(29);
    let graphs = vec![
        path(6),
        cycle(7),
        complete(5),
        grid(3, 3),
        star(8),
        disjoint_cliques(&[3, 3]),
        gnp(10, 0.4, &mut rng),
    ];
    let mut saw_split = false;
    for g in graphs {
        let c = mpx_cluster(&g, 0.25, &mut rng).expect("mpx");
        let mut seen = vec![false; g.n()];
        for cluster in c.clusters() {
            assert!(!cluster.is_empty());
            for &v in &cluster {
                assert!(!seen[v], "clusters overlap at {v}");
                seen[v] = true;
            }
        }
        for v in 0..g.n() {
            assert!(c.cluster_of(v).is_some(), "MPX leaves {v} unclustered");
        }
        assert!(
            c.strong_diameter(&g).is_some(),
            "MPX cluster not connected in G[C]"
        );
        if let Some(wd) = c.weak_diameter(&g) {
            assert!(wd <= g.n().saturating_sub(1));
        }
        if c.clusters().len() > 1 || g.n() <= 1 {
            saw_split = true;
        }
    }
    // Disjoint cliques cannot form a single MPX cluster.
    assert!(saw_split);
}

/// Definition 2 / Theorem 3 layered decomposition.
#[test]
fn theorem3_layered_invariants() {
    let mut rng = rng_seed(31);
    let graphs = vec![
        path(8),
        cycle(9),
        complete(6),
        grid(3, 3),
        star(7),
        disjoint_cliques(&[4, 3]),
        gnp(11, 0.3, &mut rng),
    ];
    for (i, g) in graphs.into_iter().enumerate() {
        let mut rng = rng_seed(100 + i as u64);
        let max_layers = ((20.0 * (g.n().max(2) as f64).ln()) as usize).max(8);
        let decomp = layered_decompose(&g, EPS, BETA, max_layers, &mut rng).expect("decomp");

        let orig: std::collections::HashSet<(usize, usize)> = g.edges().iter().copied().collect();
        let mut assigned = std::collections::HashSet::new();
        for (li, layer) in decomp.layers.iter().enumerate() {
            let gi = Graph::from_edges(g.n(), &layer.remaining).expect("Gi");
            assert!(
                layer.clustering.introvert_ok(&gi, EPS),
                "layer {li} introvert w.r.t. Gi failed"
            );

            let recomputed: Vec<(usize, usize)> = layer
                .remaining
                .iter()
                .copied()
                .filter(|&(u, v)| layer.clustering.is_intra_cluster(u, v))
                .collect();
            assert_eq!(
                recomputed, layer.intra_edges,
                "layer {li}: Ei must be intra-cluster edges of Ci in Gi"
            );

            for e in &layer.intra_edges {
                assert!(orig.contains(e), "intra edge {e:?} not in G");
                assert!(assigned.insert(*e), "edge {e:?} assigned twice");
            }
            for e in &layer.remaining {
                assert!(orig.contains(e));
            }
        }
        for e in &decomp.leftover {
            assert!(orig.contains(e));
            assert!(assigned.insert(*e), "leftover edge {e:?} already assigned");
        }
        assert_eq!(assigned.len(), orig.len(), "edge partition missing edges");
        assert!(
            decomp.leftover.is_empty(),
            "leftover nonempty on small graph: {} edges",
            decomp.leftover.len()
        );
    }

    // Several extra seeds: leftover empty.
    let g = cycle(10);
    for seed in 1..8u64 {
        let mut rng = rng_seed(seed);
        let decomp = layered_decompose(&g, EPS, BETA, 40, &mut rng).expect("decomp");
        assert!(
            decomp.leftover.is_empty(),
            "seed {seed} leftover {}",
            decomp.leftover.len()
        );
    }
}

#[test]
fn empty_single_and_edgeless() {
    let mut rng = rng_seed(3);
    for g in [Graph::new(0), Graph::new(1), Graph::from_edges(5, &[]).unwrap()] {
        let c = introvert_cluster(&g, EPS, BETA, &mut rng).expect("cluster");
        assert!(c.introvert_ok(&g, EPS));
        assert_eq!(c.inter_cluster_edge_count(&g), 0);
        let d = layered_decompose(&g, EPS, BETA, 10, &mut rng).expect("decomp");
        assert!(d.leftover.is_empty());
        assert!(d.layers.is_empty());
        let mpx = mpx_cluster(&g, 0.5, &mut rng).expect("mpx");
        assert_eq!(mpx.n(), g.n());
        let t = trim(&g, &mpx, EPS).expect("trim");
        assert!(t.introvert_ok(&g, EPS));
    }
}

#[test]
fn parameter_validation() {
    let g = cycle(4);
    let mut rng = rng_seed(1);
    assert!(validate_eps(0.0).is_err());
    assert!(validate_eps(0.5).is_err());
    assert!(validate_eps(-0.1).is_err());
    assert!(validate_eps(1.0).is_err());
    assert!(validate_eps(0.25).is_ok());
    assert!(validate_beta(0.0).is_err());
    assert!(validate_beta(1.0).is_err());
    assert!(validate_beta(1.5).is_err());
    assert!(validate_beta(-0.2).is_err());
    assert!(validate_beta(0.5).is_ok());
    assert!(validate_lambda(0.0).is_err());
    assert!(validate_lambda(-1.0).is_err());
    assert!(validate_lambda(1.1).is_err());
    assert!(validate_lambda(1.0).is_ok());
    assert!(validate_lambda(0.3).is_ok());

    assert!(mpx_cluster(&g, 0.0, &mut rng).is_err());
    assert!(mpx_cluster(&g, 1.5, &mut rng).is_err());
    assert!(trim(&g, &all_one_cluster(g.n()), 0.0).is_err());
    assert!(trim(&g, &all_one_cluster(g.n()), 0.5).is_err());
    assert!(introvert_cluster(&g, 0.0, BETA, &mut rng).is_err());
    assert!(introvert_cluster(&g, EPS, 0.0, &mut rng).is_err());
    assert!(introvert_cluster(&g, EPS, 1.0, &mut rng).is_err());
    assert!(layered_decompose(&g, 0.6, BETA, 3, &mut rng).is_err());
    assert!(layered_decompose(&g, EPS, 0.0, 3, &mut rng).is_err());
}

/// Trimming does not add vertices or merge clusters: each remaining cluster
/// is a subset of some original cluster.
#[test]
fn trim_subsets_original_clusters() {
    let mut rng = rng_seed(41);
    for g in sample_graphs(&mut rng) {
        let c0 = if g.n() == 0 {
            Clustering::new(0)
        } else {
            mpx_cluster(&g, 0.2, &mut rng).expect("mpx")
        };
        let c1 = trim(&g, &c0, EPS).expect("trim");
        for v in 0..g.n() {
            if let Some(id) = c1.cluster_of(v) {
                assert_eq!(c0.cluster_of(v), Some(id));
            }
        }
        for cluster in c1.clusters() {
            let orig_id = c1.cluster_of(cluster[0]);
            for &v in &cluster {
                assert_eq!(c0.cluster_of(v), orig_id);
            }
        }
        assert!(c1.n() <= c0.n() || g.n() == 0);
        let clustered1 = (0..g.n()).filter(|&v| c1.cluster_of(v).is_some()).count();
        let clustered0 = (0..g.n()).filter(|&v| c0.cluster_of(v).is_some()).count();
        assert!(clustered1 <= clustered0);
    }
}

/// Previously inter-cluster edges stay inter-cluster after trim.
#[test]
fn trim_does_not_repair_cut_edges() {
    let mut rng = rng_seed(43);
    for g in sample_graphs(&mut rng) {
        let c0 = if g.n() == 0 {
            Clustering::new(0)
        } else {
            mpx_cluster(&g, 0.35, &mut rng).expect("mpx")
        };
        let before: Vec<(usize, usize)> = g
            .edges()
            .iter()
            .copied()
            .filter(|&(u, v)| !c0.is_intra_cluster(u, v))
            .collect();
        let c1 = trim(&g, &c0, EPS).expect("trim");
        for (u, v) in before {
            assert!(
                !c1.is_intra_cluster(u, v),
                "trim turned inter-cluster {u}-{v} into intra-cluster"
            );
        }
        let u0 = c0.inter_cluster_edge_count(&g);
        let u1 = c1.inter_cluster_edge_count(&g);
        assert!(u1 >= u0);
    }
}
