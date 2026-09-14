use sca_scss::distance::{d_h, d_h_unweighted, layered_at_most, INF};
use sca_scss::reduce::{build_unweighted, kernelize_unweighted};
use sca_scss::{sca_unweighted, sca_weighted, terminal_count, DiGraph};

fn brute_unweighted(g: &DiGraph, links: &[(usize, usize)], k: usize) -> bool {
    if g.is_strongly_connected() {
        return true;
    }
    let p = links.len();
    assert!(p <= 20, "too many links for brute force");
    let limit = 1u32 << p;
    for mask in 0..limit {
        if mask.count_ones() as usize > k {
            continue;
        }
        let extra: Vec<(usize, usize)> = (0..p)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| links[i])
            .collect();
        if g.with_edges(&extra).is_strongly_connected() {
            return true;
        }
    }
    false
}

#[test]
fn trivial_yes_two_node() {
    let mut g = DiGraph::new(2);
    g.add_edge(0, 1);
    let links = [(1, 0)];
    assert!(sca_unweighted(&g, &links, 1));
}

#[test]
fn trivial_no_two_node_k0() {
    let mut g = DiGraph::new(2);
    g.add_edge(0, 1);
    let links = [(1, 0)];
    assert!(!sca_unweighted(&g, &links, 0));
}

#[test]
fn single_vertex_already_strong() {
    let g = DiGraph::new(1);
    assert!(sca_unweighted(&g, &[], 0));
    assert!(sca_unweighted(&g, &[], 1));
    assert!(sca_weighted(&g, &[], 0, 0));
}

#[test]
fn empty_graph_already_strong() {
    let g = DiGraph::new(0);
    assert!(sca_unweighted(&g, &[], 0));
}

#[test]
fn three_node_chain() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0)];
    assert!(sca_unweighted(&g, &links, 1));
    assert!(!sca_unweighted(&g, &[], 10));
}

#[test]
fn two_disjoint_edges() {
    let mut g = DiGraph::new(4);
    g.add_edge(0, 1);
    g.add_edge(2, 3);
    let cycle_links = [(1, 2), (3, 0)];
    assert!(sca_unweighted(&g, &cycle_links, 2));
    assert!(!sca_unweighted(&g, &cycle_links, 1));
    let split_links = [(1, 0), (3, 2)];
    assert!(!sca_unweighted(&g, &split_links, 2));
}

#[test]
fn three_sources_star_early_reject() {
    let mut g = DiGraph::new(4);
    g.add_edge(0, 3);
    g.add_edge(1, 3);
    g.add_edge(2, 3);
    assert_eq!(g.sources().len(), 3);
    assert_eq!(g.sinks().len(), 1);
    let links = [(3, 0), (3, 1), (3, 2)];
    assert!(!sca_unweighted(&g, &links, 1));
    assert!(sca_unweighted(&g, &links, 3));
}

#[test]
fn layered_distance_hand_values() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0, 5), (2, 1, 1), (1, 0, 1)];
    assert_eq!(d_h(&g, &links, 2, 0, 0), None);
    assert_eq!(d_h(&g, &links, 2, 0, 1), Some(5));
    assert_eq!(d_h(&g, &links, 2, 0, 2), Some(2));
    let table = layered_at_most(&g, &links, 2, 2);
    assert!(table[0][0] >= INF);
    assert_eq!(table[1][0], 5);
    assert_eq!(table[2][0], 2);
    assert_eq!(d_h_unweighted(&g, &[(2, 0), (2, 1), (1, 0)], 2, 0), Some(1));
}

#[test]
fn weighted_cheap_two_link_vs_expensive_one() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0, 5), (2, 1, 1), (1, 0, 1)];
    assert!(sca_weighted(&g, &links, 1, 5));
    assert!(!sca_weighted(&g, &links, 1, 4));
    assert!(sca_weighted(&g, &links, 2, 2));
    assert!(!sca_weighted(&g, &links, 2, 1));
    assert!(!sca_weighted(&g, &links, 0, 100));
    assert!(sca_unweighted(&g, &[(2, 0), (2, 1), (1, 0)], 1));
}

#[test]
fn reduction_size_at_most_two_k() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0)];
    let k = 1;
    assert!(g.sources().len() <= k);
    assert!(g.sinks().len() <= k);
    assert!(sca_unweighted(&g, &links, k));
    let z = terminal_count(&g);
    let j = build_unweighted(&g, &links);
    assert_eq!(j.n, z);
    assert!(j.n <= 2 * k);
}

#[test]
fn random_small_dags_agree_with_brute() {
    let mut rng = SplitMix64(0xC0FFEE_u64);
    let mut compared = 0;
    for trial in 0..20 {
        let n = 3 + (rng.next() as usize % 4);
        let k = 1 + (rng.next() as usize % 3);
        let mut g = DiGraph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.next() % 10 < 4 {
                    g.add_edge(i, j);
                }
            }
        }
        let mut links = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if rng.next() % 10 < 3 {
                    links.push((i, j));
                }
            }
        }
        if links.len() > 12 {
            links.truncate(12);
        }
        let algo = sca_unweighted(&g, &links, k);
        let brute = brute_unweighted(&g, &links, k);
        assert_eq!(
            algo, brute,
            "trial {trial} n={n} k={k} links={links:?} algo={algo} brute={brute}"
        );
        compared += 1;
        if g.sources().len() <= k && g.sinks().len() <= k {
            if let Some((kg, kl)) = kernelize_unweighted(&g, &links, k) {
                let kern = sca_unweighted(&kg, &kl, k);
                assert_eq!(kern, algo, "kernel mismatch trial {trial}");
            }
        }
    }
    assert_eq!(compared, 20);
}

#[test]
fn kernel_matches_small_hand_instances() {
    let mut g = DiGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    let links = [(2, 0)];
    let orig = sca_unweighted(&g, &links, 1);
    let (kg, kl) = kernelize_unweighted(&g, &links, 1).unwrap();
    assert_eq!(sca_unweighted(&kg, &kl, 1), orig);
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
