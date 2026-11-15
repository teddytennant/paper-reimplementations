//! Tests that exercise the claims of Etgar, Gilbert, Liu, McGregor,
//! arXiv:2609.15848.

use rand::rngs::StdRng;
use rand::SeedableRng;

use threshold_triangles::{
    eta, exact_triangle_count, independent_edge_estimate, random_order_edges, tau, theorem11_q,
    theorem11_q_with_c0, threshold_estimate, Graph, THEOREM11_C0,
};

fn rng() -> StdRng {
    StdRng::seed_from_u64(260915848)
}

fn complete(n: u32) -> Graph {
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            edges.push((u, v));
        }
    }
    Graph::from_edges(edges)
}

fn bipartite_complete(left: u32, right: u32) -> Graph {
    let mut edges = Vec::new();
    for u in 0..left {
        for v in 0..right {
            edges.push((u, left + v));
        }
    }
    Graph::from_edges(edges)
}

fn two_disjoint_triangles() -> Graph {
    Graph::from_edges([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
}

#[test]
fn exact_t_k3_is_1() {
    assert_eq!(exact_triangle_count(&complete(3)), 1);
}

#[test]
fn exact_t_k4_is_4() {
    assert_eq!(exact_triangle_count(&complete(4)), 4);
}

#[test]
fn exact_t_triangle_free_bipartite_is_0() {
    assert_eq!(exact_triangle_count(&bipartite_complete(2, 3)), 0);
    assert_eq!(exact_triangle_count(&bipartite_complete(3, 4)), 0);
}

#[test]
fn exact_t_two_disjoint_triangles_is_2() {
    assert_eq!(exact_triangle_count(&two_disjoint_triangles()), 2);
}

#[test]
fn closing_a_wedge_increments_by_one() {
    let mut g = Graph::new();
    g.insert(0, 1);
    g.insert(0, 2);
    assert_eq!(
        g.common_neighbor_count(1, 2),
        1,
        "inserting the closing edge of a wedge completes 1 triangle"
    );
    g.insert(1, 2);
    assert_eq!(exact_triangle_count(&g), 1);
}

#[test]
fn closing_an_edge_that_completes_k_triangles_increments_by_k() {
    // Three wedges at {0,1}: 0-2-1, 0-3-1, 0-4-1. Inserting {0,1} closes 3.
    let mut g = Graph::new();
    for w in 2..5 {
        g.insert(0, w);
        g.insert(1, w);
    }
    assert_eq!(g.common_neighbor_count(0, 1), 3);
    g.insert(0, 1);
    assert_eq!(exact_triangle_count(&g), 3);
}

#[test]
fn q_eq_1_on_a_triangle_gives_s_eq_3_and_estimate_1() {
    let edges = [(0, 1), (1, 2), (0, 2)];
    let r = threshold_estimate(&edges, 3, 1);
    assert_eq!(r.s, 3);
    assert_eq!(r.observed, 1);
    assert!(
        (r.estimate - 1.0).abs() < 1e-12,
        "T̂ = 1 · (3/3)^3 = 1, got {}",
        r.estimate
    );
}

#[test]
fn q_greater_than_t_reads_whole_stream_and_returns_exact_t() {
    let triangle = [(0, 1), (1, 2), (0, 2)];
    let r = threshold_estimate(&triangle, 3, 100);
    assert_eq!(r.s, 3, "must consume the whole stream");
    assert_eq!(r.observed, 1);
    assert!(
        (r.estimate - 1.0).abs() < 1e-12,
        "end-of-stream rule: T̂ = t̂ = T, got {}",
        r.estimate
    );

    let k4 = complete(4);
    let r = threshold_estimate(k4.edges(), k4.edge_count(), 10_000);
    assert_eq!(r.s, 6);
    assert_eq!(r.observed, 4);
    assert!((r.estimate - 4.0).abs() < 1e-12);

    let free = bipartite_complete(2, 3);
    let r = threshold_estimate(free.edges(), free.edge_count(), 1);
    assert_eq!(r.s, free.edge_count());
    assert_eq!(r.observed, 0);
    assert_eq!(r.estimate, 0.0);
}

#[test]
fn empty_stream_does_not_divide_by_s_eq_0() {
    let r = threshold_estimate(&[], 0, 1);
    assert_eq!(r.s, 0);
    assert_eq!(r.observed, 0);
    assert_eq!(r.estimate, 0.0);
}

#[test]
fn independent_edge_estimator_mean_near_t_on_k4() {
    let g = complete(4);
    let t = exact_triangle_count(&g) as f64;
    assert_eq!(t, 4.0);
    let p = 0.5;
    let trials = 8000usize;
    let mut rng = rng();
    let mut sum = 0.0;
    for _ in 0..trials {
        sum += independent_edge_estimate(&g, p, &mut rng);
    }
    let mean = sum / trials as f64;
    assert!(
        (mean - t).abs() < 0.20 * t,
        "MC mean {mean} vs T={t} (p={p}, {trials} trials)"
    );
}

#[test]
fn eta_on_kn_is_n_minus_2() {
    for n in 2..8u32 {
        let g = complete(n);
        let expected = if n >= 2 { (n - 2) as u64 } else { 0 };
        assert_eq!(eta(&g), expected, "η(K_{n})");
        for &(u, v) in g.edges() {
            assert_eq!(tau(&g, u, v), expected, "τ({u},{v}) on K_{n}");
        }
        let t = n as u64 * (n as u64 - 1) * (n as u64 - 2) / 6;
        assert_eq!(exact_triangle_count(&g), t);
    }
}

#[test]
fn prefix_estimator_uses_q_not_overshooting_t_hat() {
    // Six wedge-edges around {0,1}, then {0,1} closes 3 triangles at once.
    // Q = 2, so t̂ = 3 > Q on the last insertion. The paper returns
    // T̂ = Q (m/S)^3, not t̂ (m/S)^3.
    let stream = [(0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (0, 1)];
    let m = 10usize;
    let q = 2u64;
    let r = threshold_estimate(&stream, m, q);
    assert_eq!(r.observed, 3, "last edge closes 3 triangles");
    assert!(r.observed > q);
    assert_eq!(r.s, 7);
    let expected = (q as f64) * (m as f64 / 7.0).powi(3);
    assert!(
        (r.estimate - expected).abs() < 1e-12,
        "T̂ should use Q, got {}",
        r.estimate
    );
    let wrong = (r.observed as f64) * (m as f64 / 7.0).powi(3);
    assert!(
        (r.estimate - wrong).abs() > 1e-9,
        "must not use observed t̂ in the cubic formula"
    );
}

#[test]
fn deterministic_given_stream_order() {
    let edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)];
    let a = threshold_estimate(&edges, 6, 2);
    let b = threshold_estimate(&edges, 6, 2);
    assert_eq!(a, b);
    assert_eq!(a.s, b.s);
    assert_eq!(a.observed, b.observed);
}

#[test]
fn stopping_time_is_min_t_with_n_t_geq_q() {
    let edges = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)];
    // Prefix triangle counts N(t):
    // e1 {0,1}: 0
    // e2 {0,2}: 0
    // e3 {1,2}: 1  (closes 012)
    // e4 {0,3}: 1
    // e5 {1,3}: 2  (closes 013)
    // e6 {2,3}: 4  (closes 023 and 123)
    let q = 2u64;
    let r = threshold_estimate(&edges, 6, q);
    assert_eq!(r.s, 5, "S = min{{t : N(t) ≥ 2}} = 5");
    assert_eq!(r.observed, 2);
    let expected = 2.0f64 * (6.0f64 / 5.0).powi(3);
    assert!((r.estimate - expected).abs() < 1e-12);
}

#[test]
fn random_order_stream_then_threshold() {
    let g = complete(5);
    let mut rng = rng();
    let stream = random_order_edges(g.edges(), &mut rng);
    assert_eq!(stream.len(), 10);
    let r = threshold_estimate(&stream, 10, 1);
    assert!(r.s >= 3, "a triangle needs at least 3 edges");
    assert!(r.s <= 10);
    assert!(r.observed >= 1);
}

#[test]
fn theorem11_q_is_positive_and_scales() {
    let q = theorem11_q(0.5, 0.1);
    assert!(q > 0.0);
    let q2 = theorem11_q(0.25, 0.1);
    assert!(q2 > q, "smaller ε should give larger Q");
    let with = theorem11_q_with_c0(0.5, 0.1, THEOREM11_C0);
    assert!((q - with).abs() < 1e-12);
    let eight = theorem11_q_with_c0(0.5, 0.1, 8.0);
    assert!((eight - 8.0 * q).abs() < 1e-9);
}
