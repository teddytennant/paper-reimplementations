//! Integration tests against brute-force enumeration of live-edge worlds.

use exact_greedy_im::{exact_greedy, influence, influence_and_gradient, Graph, TreeDecomposition};

const EPS: f64 = 1e-9;

fn assert_close(a: f64, b: f64, ctx: &str) {
    assert!(
        (a - b).abs() < EPS,
        "{ctx}: {a} vs {b} (diff {})",
        (a - b).abs()
    );
}

/// Exact `σ(S)` by enumerating all `2^{|E|}` live-edge realizations (`|E| ≤ 12`).
fn brute_sigma(graph: &Graph, seeds: &[usize]) -> f64 {
    let n = graph.n();
    let edges = graph.edges();
    let m = edges.len();
    assert!(m <= 12, "brute force is only for |E| ≤ 12");
    let mut seed = vec![false; n];
    for &s in seeds {
        seed[s] = true;
    }
    let worlds = 1u32 << m;
    let mut total = 0.0;
    for mask in 0..worlds {
        let mut p = 1.0;
        let mut adj = vec![vec![]; n];
        for i in 0..m {
            if mask & (1 << i) != 0 {
                p *= edges[i].p;
                adj[edges[i].from].push(edges[i].to);
            } else {
                p *= 1.0 - edges[i].p;
            }
        }
        let mut seen = seed.clone();
        let mut stack: Vec<usize> = (0..n).filter(|&v| seed[v]).collect();
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        let reachable = seen.iter().filter(|&&b| b).count() as f64;
        total += p * reachable;
    }
    total
}

fn classical_greedy_brute(graph: &Graph, k: usize) -> Vec<usize> {
    let n = graph.n();
    let mut s = Vec::new();
    for _ in 0..k {
        let sigma_s = brute_sigma(graph, &s);
        let mut best_v = None;
        let mut best_d = f64::NEG_INFINITY;
        for v in 0..n {
            if s.contains(&v) {
                continue;
            }
            let mut s2 = s.clone();
            s2.push(v);
            let d = brute_sigma(graph, &s2) - sigma_s;
            if best_v.is_none() || d > best_d {
                best_d = d;
                best_v = Some(v);
            }
        }
        s.push(best_v.unwrap());
    }
    s
}

fn theta_of(n: usize, seeds: &[usize]) -> Vec<f64> {
    let mut t = vec![0.0; n];
    for &s in seeds {
        t[s] = 1.0;
    }
    t
}

fn check_sigma_and_grad(graph: &Graph, td: &TreeDecomposition, seeds: &[usize]) {
    td.validate(graph).expect("TD");
    let n = graph.n();
    let brute = brute_sigma(graph, seeds);
    let dp = influence(graph, td, seeds);
    assert_close(dp, brute, &format!("σ({seeds:?})"));

    let theta = theta_of(n, seeds);
    let (f, grad) = influence_and_gradient(graph, td, &theta);
    assert_close(f, brute, &format!("F(1_{seeds:?})"));

    for v in 0..n {
        if seeds.contains(&v) {
            continue;
        }
        let mut s2 = seeds.to_vec();
        s2.push(v);
        let marg = brute_sigma(graph, &s2) - brute;
        assert_close(grad[v], marg, &format!("∂F/∂θ_{v} at 1_{seeds:?}"));
    }
}

fn directed_path(n: usize, p: f64) -> (Graph, TreeDecomposition) {
    let mut g = Graph::new(n);
    for i in 0..n - 1 {
        g.add_edge(i, i + 1, p);
    }
    (g, TreeDecomposition::path(n))
}

fn small_tree() -> (Graph, TreeDecomposition) {
    //     0
    //    / \
    //   1   2
    //  /
    // 3
    let mut g = Graph::new(4);
    g.add_edge(0, 1, 0.4);
    g.add_edge(0, 2, 0.7);
    g.add_edge(1, 3, 0.5);
    let td = TreeDecomposition::for_tree(4, &[(0, 1), (0, 2), (1, 3)]);
    (g, td)
}

fn cycle4_chord() -> (Graph, TreeDecomposition) {
    // 0 → 1 → 2 → 3 → 0 and chord 0 → 2
    let mut g = Graph::new(4);
    g.add_edge(0, 1, 0.3);
    g.add_edge(1, 2, 0.6);
    g.add_edge(2, 3, 0.5);
    g.add_edge(3, 0, 0.4);
    g.add_edge(0, 2, 0.2);
    (g, TreeDecomposition::cycle4_with_chord())
}

#[test]
fn path_matches_brute() {
    let (g, td) = directed_path(5, 0.5);
    check_sigma_and_grad(&g, &td, &[]);
    check_sigma_and_grad(&g, &td, &[0]);
    check_sigma_and_grad(&g, &td, &[2]);
    check_sigma_and_grad(&g, &td, &[4]);
    check_sigma_and_grad(&g, &td, &[0, 3]);
}

#[test]
fn path_random_p() {
    let mut g = Graph::new(4);
    let ps = [0.13, 0.77, 0.41];
    for i in 0..3 {
        g.add_edge(i, i + 1, ps[i]);
    }
    let td = TreeDecomposition::path(4);
    check_sigma_and_grad(&g, &td, &[]);
    check_sigma_and_grad(&g, &td, &[1]);
    check_sigma_and_grad(&g, &td, &[0, 2]);
}

#[test]
fn tree_matches_brute() {
    let (g, td) = small_tree();
    check_sigma_and_grad(&g, &td, &[]);
    check_sigma_and_grad(&g, &td, &[0]);
    check_sigma_and_grad(&g, &td, &[1]);
    check_sigma_and_grad(&g, &td, &[3]);
    check_sigma_and_grad(&g, &td, &[2, 3]);
}

#[test]
fn cycle_with_chord_matches_brute() {
    let (g, td) = cycle4_chord();
    check_sigma_and_grad(&g, &td, &[]);
    check_sigma_and_grad(&g, &td, &[0]);
    check_sigma_and_grad(&g, &td, &[1]);
    check_sigma_and_grad(&g, &td, &[2]);
    check_sigma_and_grad(&g, &td, &[0, 2]);
    check_sigma_and_grad(&g, &td, &[1, 3]);
}

#[test]
fn greedy_k2_matches_classical() {
    // Asymmetric path so the two greedy steps are unique.
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 0.8);
    g.add_edge(1, 2, 0.5);
    let td = TreeDecomposition::path(3);
    let exact = exact_greedy(&g, &td, 2);
    let classical = classical_greedy_brute(&g, 2);
    assert_eq!(exact, classical);
    assert_eq!(exact, vec![0, 2]);
}

#[test]
fn unique_ownership_edge_in_two_bags() {
    // Path 0 → 1 → 2. Edge 0→1 sits in both {0,1,2} and {0,1}; 1→2 sits in
    // both {0,1,2} and {1,2}. Ownership assigns each edge to the first bag
    // that contains both endpoints, so it is sampled once.
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 0.5);
    g.add_edge(1, 2, 0.5);
    let td = TreeDecomposition::new(
        vec![vec![0, 1, 2], vec![0, 1], vec![1, 2]],
        vec![(0, 1), (0, 2)],
    );
    td.validate(&g).unwrap();
    check_sigma_and_grad(&g, &td, &[]);
    check_sigma_and_grad(&g, &td, &[0]);
    check_sigma_and_grad(&g, &td, &[1]);
    check_sigma_and_grad(&g, &td, &[0, 1]);
}

#[test]
fn ad_keeps_both_branches_at_theta_zero() {
    // At θ = 0, pruning the θ_v branch would make every gradient 0.
    let (g, td) = directed_path(4, 0.6);
    let theta = vec![0.0; 4];
    let (f, grad) = influence_and_gradient(&g, &td, &theta);
    assert_close(f, 0.0, "F(0)");
    for v in 0..4 {
        let marg = brute_sigma(&g, &[v]);
        assert!(
            grad[v].abs() > 0.5,
            "gradient at θ=0 vanished for v={v}: {}",
            grad[v]
        );
        assert_close(grad[v], marg, &format!("∂F/∂θ_{v} at 0"));
    }
}
