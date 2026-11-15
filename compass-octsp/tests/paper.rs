//! Paper-faithful tests for COMPASS on Ordered Clustered TSP (arXiv:2609.20352).
//! Identities, not smoke checks.

use compass_octsp::{
    candidate_pairs, compass, compass_exact, connect_first, held_karp_path, OctspInstance,
};

const EPS: f64 = 1e-8;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs().max(b.abs()))
}

fn path_cost(dist: &[Vec<f64>], tour: &[usize]) -> f64 {
    tour.windows(2).map(|w| dist[w[0]][w[1]]).sum()
}

fn permute<T: Clone>(items: &mut [T], start: usize, out: &mut Vec<Vec<T>>) {
    if start == items.len() {
        out.push(items.to_vec());
        return;
    }
    for i in start..items.len() {
        items.swap(start, i);
        permute(items, start + 1, out);
        items.swap(start, i);
    }
}

fn brute_fixed_endpoint(
    dist: &[Vec<f64>],
    nodes: &[usize],
    s: usize,
    t: usize,
) -> (f64, Vec<usize>) {
    if s == t {
        if nodes.len() == 1 && nodes[0] == s {
            return (0.0, vec![s]);
        }
        return (f64::INFINITY, vec![]);
    }
    if !nodes.contains(&s) || !nodes.contains(&t) {
        return (f64::INFINITY, vec![]);
    }
    let middle: Vec<usize> = nodes
        .iter()
        .copied()
        .filter(|&u| u != s && u != t)
        .collect();
    let mut perms = Vec::new();
    let mut mid = middle;
    permute(&mut mid, 0, &mut perms);
    if perms.is_empty() {
        perms.push(vec![]);
    }
    let mut best = f64::INFINITY;
    let mut best_path = Vec::new();
    for perm in perms {
        let mut path = Vec::with_capacity(nodes.len());
        path.push(s);
        path.extend(perm);
        path.push(t);
        let c = path_cost(dist, &path);
        if c < best {
            best = c;
            best_path = path;
        }
    }
    (best, best_path)
}

fn brute_octsp(inst: &OctspInstance) -> (f64, Vec<usize>) {
    let mut best_cost = f64::INFINITY;
    let mut best_tour = Vec::new();
    let mut prefix = vec![inst.source];
    brute_octsp_rec(inst, 0, &mut prefix, &mut best_cost, &mut best_tour);
    (best_cost, best_tour)
}

fn brute_octsp_rec(
    inst: &OctspInstance,
    i: usize,
    prefix: &mut Vec<usize>,
    best_cost: &mut f64,
    best_tour: &mut Vec<usize>,
) {
    if i == inst.clusters.len() {
        prefix.push(inst.target);
        let c = path_cost(&inst.distances, prefix);
        if c < *best_cost {
            *best_cost = c;
            *best_tour = prefix.clone();
        }
        prefix.pop();
        return;
    }
    let mut nodes = inst.clusters[i].clone();
    let mut perms = Vec::new();
    permute(&mut nodes, 0, &mut perms);
    for perm in perms {
        let mark = prefix.len();
        prefix.extend_from_slice(&perm);
        brute_octsp_rec(inst, i + 1, prefix, best_cost, best_tour);
        prefix.truncate(mark);
    }
}

fn tour_visits_clusters_in_order(inst: &OctspInstance, tour: &[usize]) -> bool {
    if tour.first().copied() != Some(inst.source) {
        return false;
    }
    if tour.last().copied() != Some(inst.target) {
        return false;
    }
    let mut pos = 1usize;
    let mut seen = vec![false; inst.distances.len()];
    if inst.source < seen.len() {
        seen[inst.source] = true;
    }
    for cluster in &inst.clusters {
        let mut remaining: Vec<usize> = cluster.clone();
        for _ in 0..cluster.len() {
            if pos >= tour.len() - 1 {
                return false;
            }
            let v = tour[pos];
            if let Some(idx) = remaining.iter().position(|&u| u == v) {
                remaining.swap_remove(idx);
            } else {
                return false;
            }
            if v >= seen.len() || seen[v] {
                return false;
            }
            seen[v] = true;
            pos += 1;
        }
    }
    pos == tour.len() - 1
}

/// Asymmetric 5-node complete matrix used as a single cluster.
fn asymmetric5() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 2.0, 9.0, 10.0, 1.5],
        vec![3.0, 0.0, 4.0, 6.0, 8.0],
        vec![7.0, 5.0, 0.0, 2.0, 3.0],
        vec![4.0, 9.0, 1.0, 0.0, 5.0],
        vec![6.0, 2.5, 8.0, 7.0, 0.0],
    ]
}

/// Tiny OCTSP: S=0, C1={1,2}, C2={3,4}, T=5. Asymmetric.
fn tiny_two_clusters() -> OctspInstance {
    let n = 6;
    let mut d = vec![vec![0.0; n]; n];
    let entries = [
        (0, 1, 4.0),
        (0, 2, 1.0),
        (0, 3, 9.0),
        (0, 4, 9.0),
        (0, 5, 20.0),
        (1, 0, 5.0),
        (1, 2, 3.0),
        (1, 3, 2.0),
        (1, 4, 8.0),
        (1, 5, 9.0),
        (2, 0, 2.0),
        (2, 1, 6.0),
        (2, 3, 7.0),
        (2, 4, 1.5),
        (2, 5, 9.0),
        (3, 0, 8.0),
        (3, 1, 3.0),
        (3, 2, 6.0),
        (3, 4, 2.5),
        (3, 5, 4.0),
        (4, 0, 8.0),
        (4, 1, 7.0),
        (4, 2, 2.0),
        (4, 3, 5.0),
        (4, 5, 1.0),
        (5, 0, 20.0),
        (5, 1, 9.0),
        (5, 2, 9.0),
        (5, 3, 3.0),
        (5, 4, 2.0),
    ];
    for (i, j, w) in entries {
        d[i][j] = w;
    }
    OctspInstance {
        distances: d,
        clusters: vec![vec![1, 2], vec![3, 4]],
        source: 0,
        target: 5,
    }
}

/// Three clusters, mixed sizes: S=0, C1={1,2}, C2={3,4,5}, C3={6,7}, T=8.
fn tiny_three_clusters() -> OctspInstance {
    let n = 9;
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Asymmetric, deterministic, strictly positive.
            d[i][j] = ((i * 7 + j * 13) % 17) as f64 + 1.0 + if i > j { 0.5 } else { 0.0 };
        }
    }
    OctspInstance {
        distances: d,
        clusters: vec![vec![1, 2], vec![3, 4, 5], vec![6, 7]],
        source: 0,
        target: 8,
    }
}

/// Single-node clusters: S=0, C1={1}, C2={2}, C3={3}, T=4.
fn single_node_clusters() -> OctspInstance {
    let n = 5;
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                d[i][j] = (i + 2 * j + 1) as f64;
                if i < j {
                    d[i][j] += 0.25;
                }
            }
        }
    }
    OctspInstance {
        distances: d,
        clusters: vec![vec![1], vec![2], vec![3]],
        source: 0,
        target: 4,
    }
}

fn layered_cost(inst: &OctspInstance, endpoints: &[(usize, usize)]) -> f64 {
    let mut cost = 0.0;
    let mut prev_exit = inst.source;
    for (i, &(s, t)) in endpoints.iter().enumerate() {
        cost += inst.distances[prev_exit][s];
        let (intra, _) = held_karp_path(&inst.distances, &inst.clusters[i], s, t);
        cost += intra;
        prev_exit = t;
    }
    cost += inst.distances[prev_exit][inst.target];
    cost
}

#[test]
fn held_karp_matches_brute_force_asymmetric() {
    let dist = asymmetric5();
    let nodes: Vec<usize> = vec![0, 1, 2, 3, 4];
    let pairs = [(0, 1), (0, 4), (2, 3), (4, 0), (1, 3), (3, 1), (2, 0)];
    for &(s, t) in &pairs {
        let (hk_cost, hk_path) = held_karp_path(&dist, &nodes, s, t);
        let (bf_cost, bf_path) = brute_fixed_endpoint(&dist, &nodes, s, t);
        assert!(
            close(hk_cost, bf_cost),
            "Held-Karp cost {hk_cost} != brute {bf_cost} for ({s},{t})"
        );
        assert_eq!(hk_path.len(), nodes.len());
        assert_eq!(hk_path[0], s);
        assert_eq!(*hk_path.last().unwrap(), t);
        assert_eq!(hk_path.len(), bf_path.len());
        assert!(close(path_cost(&dist, &hk_path), hk_cost));
        // Every cluster node once.
        let mut sorted = hk_path.clone();
        sorted.sort_unstable();
        let mut expect = nodes.clone();
        expect.sort_unstable();
        assert_eq!(sorted, expect);
    }
}

#[test]
fn compass_exact_matches_brute_force_octsp() {
    for inst in [tiny_two_clusters(), tiny_three_clusters()] {
        let sol = compass_exact(&inst);
        let (bf_cost, bf_tour) = brute_octsp(&inst);
        assert!(
            close(sol.cost, bf_cost),
            "compass_exact {} != brute {} tour={:?} brute={:?}",
            sol.cost,
            bf_cost,
            sol.tour,
            bf_tour
        );
        assert!(tour_visits_clusters_in_order(&inst, &sol.tour));
        assert!(close(path_cost(&inst.distances, &sol.tour), sol.cost));
    }
}

#[test]
fn greedy_bridge_pair_always_in_pi() {
    for inst in [
        tiny_two_clusters(),
        tiny_three_clusters(),
        single_node_clusters(),
    ] {
        let cf = connect_first(&inst);
        for k in [1usize, 2, 3, 8] {
            let pis = candidate_pairs(&inst, k);
            assert_eq!(pis.len(), inst.clusters.len());
            for (i, pair) in cf.endpoints.iter().enumerate() {
                assert!(
                    pis[i].contains(pair),
                    "Connect-first pair {:?} missing from P_{i} at K={k}: {:?}",
                    pair,
                    pis[i]
                );
            }
        }
    }
}

#[test]
fn connect_first_never_better_than_compass_exact() {
    for inst in [
        tiny_two_clusters(),
        tiny_three_clusters(),
        single_node_clusters(),
    ] {
        let exact = compass_exact(&inst);
        let cf = connect_first(&inst);
        assert!(
            cf.cost + EPS >= exact.cost,
            "Connect-first {} better than exact COMPASS {}",
            cf.cost,
            exact.cost
        );
        assert!(tour_visits_clusters_in_order(&inst, &cf.tour));
        assert!(tour_visits_clusters_in_order(&inst, &exact.tour));
        // Exhaustive candidates + Held-Karp is optimal (Corollary 2 / Proposition 1).
        let (bf_cost, _) = brute_octsp(&inst);
        assert!(close(exact.cost, bf_cost));
    }
}

#[test]
fn asymmetric_distances_valid_tour() {
    let inst = tiny_three_clusters();
    let mut found_asym = false;
    for i in 0..inst.distances.len() {
        for j in 0..inst.distances.len() {
            if (inst.distances[i][j] - inst.distances[j][i]).abs() > EPS {
                found_asym = true;
            }
        }
    }
    assert!(found_asym, "test matrix should be asymmetric");
    let sol = compass_exact(&inst);
    assert!(tour_visits_clusters_in_order(&inst, &sol.tour));
    let mut seen = vec![false; inst.distances.len()];
    for &v in &sol.tour {
        assert!(!seen[v], "node {v} visited twice");
        seen[v] = true;
    }
    for cluster in &inst.clusters {
        for &u in cluster {
            assert!(seen[u]);
        }
    }
    assert!(seen[inst.source] && seen[inst.target]);
    assert!(close(path_cost(&inst.distances, &sol.tour), sol.cost));
}

#[test]
fn single_node_clusters_work() {
    let inst = single_node_clusters();
    let sol = compass_exact(&inst);
    assert_eq!(sol.tour, vec![0, 1, 2, 3, 4]);
    let expect =
        inst.distances[0][1] + inst.distances[1][2] + inst.distances[2][3] + inst.distances[3][4];
    assert!(close(sol.cost, expect));
    assert_eq!(sol.endpoints, vec![(1, 1), (2, 2), (3, 3)]);
    let cf = connect_first(&inst);
    assert_eq!(cf.tour, sol.tour);
    assert!(close(cf.cost, sol.cost));
    let approx = compass(&inst, 1);
    assert_eq!(approx.tour, sol.tour);
}

#[test]
fn layered_dp_cost_equals_held_karp_plus_bridges() {
    for inst in [
        tiny_two_clusters(),
        tiny_three_clusters(),
        single_node_clusters(),
    ] {
        let sol = compass_exact(&inst);
        let decomposed = layered_cost(&inst, &sol.endpoints);
        assert!(
            close(sol.cost, decomposed),
            "DP cost {} != sum of intra Held-Karp + bridges {}",
            sol.cost,
            decomposed
        );
        assert!(close(sol.cost, path_cost(&inst.distances, &sol.tour)));
        let again = compass(&inst, 100);
        assert!(close(again.cost, sol.cost));
    }
}

#[test]
fn invalid_s_eq_t_not_a_candidate() {
    let inst = tiny_three_clusters();
    for k in [1usize, 4, 50] {
        let pis = candidate_pairs(&inst, k);
        for (i, cluster) in inst.clusters.iter().enumerate() {
            if cluster.len() > 1 {
                for &(s, t) in &pis[i] {
                    assert_ne!(s, t, "invalid s==t candidate in cluster {i} at K={k}");
                    assert!(cluster.contains(&s) && cluster.contains(&t));
                }
            } else {
                assert_eq!(pis[i], vec![(cluster[0], cluster[0])]);
            }
        }
    }
    let nodes = vec![0usize, 1, 2, 3, 4];
    let (cost, path) = held_karp_path(&asymmetric5(), &nodes, 2, 2);
    assert!(!cost.is_finite());
    assert!(path.is_empty());
}
