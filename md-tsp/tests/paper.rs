//! Tests of the claims in Zhao, Liu, Xiao, arXiv:2609.19537.

use md_tsp::{
    approximate, approximate_with_lambda, eta, min_cost_perfect_matching, odd_vertices,
    primal_dual_rsf, theta_ell, w_weight, Instance, LAMBDA, TAU,
};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs() + b.abs())
}

/// Paper Figure 1: line metric with u1, u2 at 0, 13 and clients a, b, d at 2, 3, 7.
fn figure1() -> Instance {
    Instance::from_line(&[0.0, 13.0], &[2.0, 3.0, 7.0])
}

/// One depot at the origin and two clients; metric TSP-like.
fn one_depot_toy() -> Instance {
    Instance::from_euclidean(&[[0.0, 0.0]], &[[1.0, 0.0], [0.0, 1.0]])
}

/// Two depots and three clients on a small Euclidean cloud.
fn two_depot_toy() -> Instance {
    Instance::from_euclidean(
        &[[0.0, 0.0], [4.0, 0.0]],
        &[[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]],
    )
}

fn square_one_depot() -> Instance {
    Instance::from_euclidean(&[[0.0, 0.0]], &[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
}

struct Uf {
    parent: Vec<usize>,
}

impl Uf {
    fn new(n: usize, n_depots: usize) -> Self {
        let mut parent: Vec<usize> = (0..n).collect();
        for u in 1..n_depots {
            // contract depots
            parent[u] = 0;
        }
        Self { parent }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut x = x;
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[rb] = ra;
        }
    }
}

fn kappa(
    inst: &Instance,
    lambda: f64,
    labels: &[f64],
    in_root: &[bool],
    u: usize,
    v: usize,
) -> f64 {
    match (in_root[u], in_root[v]) {
        (false, false) => inst.cost(u, v),
        (true, false) => (1.0 + lambda) * inst.cost(u, v) - lambda * labels[u],
        (false, true) => (1.0 + lambda) * inst.cost(u, v) - lambda * labels[v],
        (true, true) => f64::INFINITY,
    }
}

/// Dual contribution to the constraint of uv at time s (Section 3.1).
fn dual_contribution(s: f64, lambda: f64, rooted_label: Option<f64>) -> f64 {
    let eta = eta(lambda);
    match rooted_label {
        None => s,
        Some(a) => a + (1.0 + eta) / 2.0 * (s - a),
    }
}

fn assert_feasible(inst: &Instance, sol: &md_tsp::MdTspSolution) {
    assert_eq!(sol.tours.len(), inst.n_depots());
    let mut covered = vec![false; inst.n()];
    for (u, tour) in sol.tours.iter().enumerate() {
        assert!(!tour.is_empty(), "tour of depot {u} is empty");
        assert_eq!(tour[0], u, "tour must start at its depot");
        assert_eq!(*tour.last().unwrap(), u, "tour must end at its depot");
        assert!(inst.is_depot(u));
        if tour.len() == 1 {
            continue;
        }
        assert_eq!(
            tour[0],
            tour[tour.len() - 1],
            "nontrivial tours are closed walks"
        );
        for &v in tour {
            if !inst.is_depot(v) {
                assert!(!covered[v], "client {v} visited in two tours");
                covered[v] = true;
            }
        }
    }
    for v in inst.n_depots()..inst.n() {
        assert!(covered[v], "client {v} is not covered");
    }
}

fn assert_primal_dual_invariants(inst: &Instance, forest: &md_tsp::RootedSpanningForest) {
    let n = inst.n();
    let k = inst.n_depots();
    assert_eq!(forest.events.len(), n - k, "Lemma 1: |V| − k merges");
    for u in 0..k {
        assert!(approx_eq(forest.labels[u], 0.0), "ℓ(depot) = 0");
    }
    for v in k..n {
        assert!(forest.labels[v].is_finite(), "every client is labeled");
        assert!(forest.labels[v] >= -1e-12);
    }
    for w in forest.events.windows(2) {
        assert!(
            w[0].time <= w[1].time + 1e-12,
            "event times are nondecreasing"
        );
    }

    let mut uf = Uf::new(n, k);
    let mut in_root = vec![false; n];
    for u in 0..k {
        in_root[u] = true;
    }
    let mut assigned = forest.labels.clone();
    for v in k..n {
        assigned[v] = f64::NAN;
    }

    for (idx, ev) in forest.events.iter().enumerate() {
        assert_ne!(
            uf.find(ev.u),
            uf.find(ev.v),
            "selected edge joins distinct components"
        );
        let mut best = f64::INFINITY;
        let mut best_e = (0, 0);
        for a in 0..n {
            for b in (a + 1)..n {
                if uf.find(a) == uf.find(b) {
                    continue;
                }
                let key = kappa(inst, forest.lambda, &assigned, &in_root, a, b);
                let better = match key.partial_cmp(&best) {
                    Some(std::cmp::Ordering::Less) => true,
                    Some(std::cmp::Ordering::Greater) => false,
                    _ => (a, b) < best_e,
                };
                if better {
                    best = key;
                    best_e = (a, b);
                }
            }
        }
        assert_eq!(
            best_e,
            (ev.u.min(ev.v), ev.u.max(ev.v)),
            "event {idx}: min-key edge was not the one selected"
        );
        let (u, v) = (ev.u, ev.v);
        assert!(
            approx_eq(best, ev.time),
            "event {idx}: selected key {} vs time {}",
            best,
            ev.time
        );
        let rooted_label = match (in_root[u], in_root[v]) {
            (false, false) => None,
            (true, false) => Some(assigned[u]),
            (false, true) => Some(assigned[v]),
            (true, true) => panic!("both endpoints already in R0"),
        };
        let contrib = dual_contribution(ev.time, forest.lambda, rooted_label);
        assert!(
            approx_eq(contrib, inst.cost(u, v)),
            "event {idx}: dual contribution {contrib} != c={}",
            inst.cost(u, v)
        );

        let w = w_weight(
            inst.cost(u, v),
            forest.lambda,
            forest.labels[u],
            forest.labels[v],
        );
        assert!(
            approx_eq(w, ev.time),
            "Lemma 2(i): w(e) equals event time ({w} vs {})",
            ev.time
        );

        let joining_root = in_root[u] ^ in_root[v];
        if joining_root {
            let fresh = if in_root[u] { v } else { u };
            assert!(
                ev.newly_labeled.contains(&fresh),
                "the newly rooted side is labeled at connection time"
            );
        } else {
            assert!(
                !in_root[u] && !in_root[v],
                "the only remaining case is a rootless–rootless merge"
            );
            assert!(
                ev.newly_labeled.is_empty(),
                "rootless–rootless merge assigns no labels"
            );
        }
        for &x in &ev.newly_labeled {
            assert!(!in_root[x], "newly labeled vertex was rootless");
            assert!(approx_eq(ev.time, forest.labels[x]));
            assigned[x] = ev.time;
            in_root[x] = true;
        }
        uf.union(u, v);
    }

    for v in k..n {
        assert!(in_root[v], "Property 1: client {v} connected to a depot");
        assert!(approx_eq(assigned[v], forest.labels[v]));
    }
}

#[test]
fn instances_are_metric() {
    for inst in [
        figure1(),
        one_depot_toy(),
        two_depot_toy(),
        square_one_depot(),
    ] {
        assert!(inst.is_metric(1e-9));
    }
}

#[test]
fn tau_and_lambda_are_sqrt2_minus_1() {
    let expected = 2.0_f64.sqrt() - 1.0;
    assert!(approx_eq(LAMBDA, expected));
    assert!(approx_eq(TAU, expected));
    assert!(approx_eq(eta(LAMBDA), LAMBDA));
    assert!(approx_eq(
        md_tsp::approximation_ratio(),
        1.0 + 1.0 / 2.0_f64.sqrt()
    ));
}

#[test]
fn figure1_primal_dual_events_and_labels() {
    let inst = figure1();
    let forest = primal_dual_rsf(&inst, 0.5);
    assert_eq!(forest.events.len(), 3);
    let edges: Vec<(usize, usize, f64)> = forest
        .events
        .iter()
        .map(|e| (e.u.min(e.v), e.u.max(e.v), e.time))
        .collect();
    assert_eq!(edges[0].0, 2);
    assert_eq!(edges[0].1, 3);
    assert!(approx_eq(edges[0].2, 1.0), "select ab at s=1");
    assert_eq!(edges[1].0, 0);
    assert_eq!(edges[1].1, 2);
    assert!(approx_eq(edges[1].2, 3.0), "select u1a at s=3");
    assert_eq!(edges[2].0, 3);
    assert_eq!(edges[2].1, 4);
    assert!(approx_eq(edges[2].2, 4.5), "select bd at s=h=9/2");

    assert!(approx_eq(forest.labels[0], 0.0));
    assert!(approx_eq(forest.labels[1], 0.0));
    assert!(approx_eq(forest.labels[2], 3.0), "ℓ(a)=3");
    assert!(approx_eq(forest.labels[3], 3.0), "ℓ(b)=3");
    assert!(approx_eq(forest.labels[4], 4.5), "ℓ(d)=h");

    assert!(forest.events[0].newly_labeled.is_empty());
    let mut lab = forest.events[1].newly_labeled.clone();
    lab.sort();
    assert_eq!(lab, vec![2, 3]);
    assert_eq!(forest.events[2].newly_labeled, vec![4]);
}

#[test]
fn primal_dual_growth_adds_edge_at_tightness() {
    for (inst, lambda) in [
        (figure1(), 0.5),
        (figure1(), LAMBDA),
        (one_depot_toy(), LAMBDA),
        (two_depot_toy(), LAMBDA),
        (square_one_depot(), 0.0),
        (square_one_depot(), 1.0),
    ] {
        let forest = primal_dual_rsf(&inst, lambda);
        assert_primal_dual_invariants(&inst, &forest);
    }
}

#[test]
fn labels_are_connection_times() {
    let inst = two_depot_toy();
    let forest = primal_dual_rsf(&inst, LAMBDA);
    let mut uf = Uf::new(inst.n(), inst.n_depots());
    let mut connected = vec![false; inst.n()];
    for u in 0..inst.n_depots() {
        connected[u] = true;
    }
    for ev in &forest.events {
        let u_c = connected[ev.u];
        let v_c = connected[ev.v];
        uf.union(ev.u, ev.v);
        if u_c && v_c {
            panic!("edge inside R0");
        }
        if u_c || v_c {
            for &x in &ev.newly_labeled {
                assert!(!connected[x]);
                connected[x] = true;
                assert!(
                    approx_eq(forest.labels[x], ev.time),
                    "Property 1: ℓ(v) is the time v first meets a depot"
                );
            }
        }
    }
    for v in inst.n_depots()..inst.n() {
        assert!(connected[v]);
    }
}

#[test]
fn lemma2_selected_weights_equal_event_times_and_msf() {
    for inst in [
        figure1(),
        one_depot_toy(),
        two_depot_toy(),
        square_one_depot(),
    ] {
        let forest = primal_dual_rsf(&inst, LAMBDA);
        for ev in &forest.events {
            let w = w_weight(
                inst.cost(ev.u, ev.v),
                LAMBDA,
                forest.labels[ev.u],
                forest.labels[ev.v],
            );
            assert!(approx_eq(w, ev.time));
        }
        let w_f = forest.w_cost(&inst);
        let (_, msf) = inst.min_weight_rsf(|u, v| {
            w_weight(inst.cost(u, v), LAMBDA, forest.labels[u], forest.labels[v])
        });
        assert!(
            approx_eq(w_f, msf),
            "Lemma 2(ii): w(F) = msf_w(D) ({w_f} vs {msf})"
        );
    }
}

#[test]
fn lambda_zero_recovers_min_cost_rsf() {
    for inst in [figure1(), one_depot_toy(), two_depot_toy()] {
        let forest = primal_dual_rsf(&inst, 0.0);
        let (_, msf) = inst.min_cost_rsf();
        assert!(approx_eq(forest.cost(&inst), msf));
        for ev in &forest.events {
            assert!(approx_eq(ev.time, inst.cost(ev.u, ev.v)));
        }
    }
}

#[test]
fn one_depot_is_metric_tsp_like() {
    let inst = one_depot_toy();
    assert_eq!(inst.n_depots(), 1);
    let sol = approximate(&inst);
    assert_feasible(&inst, &sol);
    assert_eq!(sol.tours.len(), 1);
    let tour = &sol.tours[0];
    assert_eq!(tour[0], 0);
    assert_eq!(*tour.last().unwrap(), 0);
    let clients: Vec<usize> = tour
        .iter()
        .copied()
        .filter(|&v| !inst.is_depot(v))
        .collect();
    assert_eq!(clients.len(), inst.n_clients());
    let cost = sol.cost(&inst);
    assert!(cost.is_finite());
    let (_, rsf) = inst.min_cost_rsf();
    assert!(
        cost + 1e-9 >= rsf,
        "feasible cost is at least the RSF lower bound"
    );
}

#[test]
fn two_depot_toy_is_feasible_and_bounded() {
    let inst = two_depot_toy();
    assert_eq!(inst.n_depots(), 2);
    let sol = approximate(&inst);
    assert_feasible(&inst, &sol);
    let cost = sol.cost(&inst);
    assert!(cost.is_finite() && cost >= 0.0);
    let (_, rsf) = inst.min_cost_rsf();
    assert!(cost + 1e-9 >= rsf);
    let c_f = sol.forest.cost(&inst);
    let c_m: f64 = sol.matching.iter().map(|&(u, v)| inst.cost(u, v)).sum();
    assert!(cost <= c_f + c_m + 1e-9, "Eq. (2): c(T) ≤ c(F) + c(M)");
}

#[test]
fn figure1_full_algorithm() {
    let inst = figure1();
    let sol = approximate_with_lambda(&inst, 0.5);
    assert_feasible(&inst, &sol);
    let cost = sol.cost(&inst);
    let (_, rsf) = inst.min_cost_rsf();
    assert!(cost.is_finite() && cost + 1e-9 >= rsf);
}

#[test]
fn odd_f_has_even_cardinality_and_matching_covers_it() {
    for inst in [
        figure1(),
        one_depot_toy(),
        two_depot_toy(),
        square_one_depot(),
    ] {
        let sol = approximate(&inst);
        let odd = odd_vertices(inst.n(), &sol.forest.edges());
        assert_eq!(odd.len() % 2, 0);
        assert_eq!(sol.matching.len(), odd.len() / 2);
        let mut seen = vec![false; inst.n()];
        for &(u, v) in &sol.matching {
            assert!(odd.contains(&u) && odd.contains(&v));
            assert!(!seen[u] && !seen[v]);
            seen[u] = true;
            seen[v] = true;
        }
        for &v in &odd {
            assert!(seen[v]);
        }
    }
}

#[test]
fn matching_dp_agrees_with_brute_four_points() {
    let inst = square_one_depot();
    let verts = vec![0, 1, 2, 3];
    let m = min_cost_perfect_matching(&inst, &verts);
    assert_eq!(m.len(), 2);
    let cost: f64 = m.iter().map(|&(u, v)| inst.cost(u, v)).sum();
    let brute = [
        inst.cost(0, 1) + inst.cost(2, 3),
        inst.cost(0, 2) + inst.cost(1, 3),
        inst.cost(0, 3) + inst.cost(1, 2),
    ];
    let best = brute.into_iter().fold(f64::INFINITY, f64::min);
    assert!(approx_eq(cost, best));
}

#[test]
fn theta_ell_definition() {
    assert!(approx_eq(theta_ell(4.0, 3.0, 4.5), 1.0));
    assert!(approx_eq(theta_ell(1.0, 3.0, 3.0), 0.0));
}

#[test]
fn unused_depot_gets_trivial_tour() {
    let inst = Instance::from_euclidean(&[[0.0, 0.0], [100.0, 0.0]], &[[1.0, 0.0], [2.0, 0.0]]);
    let sol = approximate(&inst);
    assert_feasible(&inst, &sol);
    let trivial = sol.tours.iter().filter(|t| t.len() == 1).count();
    assert!(trivial >= 1, "far depot should remain unused");
}

#[test]
fn one_depot_square_cost_at_least_mst() {
    let inst = square_one_depot();
    let sol = approximate(&inst);
    assert_feasible(&inst, &sol);
    let (_, mst) = inst.min_cost_rsf();
    assert!(sol.cost(&inst) + 1e-9 >= mst);
    assert!(sol.cost(&inst).is_finite());
}
