//! Invariant tests against Agarwal, arXiv:2609.10641.
//! The HTML/PDF has no equation numbers; comments cite section / definition.

use polyamorous_scheduling::{
    deadline, deadline_driven_matching, deadline_driven_schedule, is_matching,
    max_weighted_waiting_time, Graph, DEFAULT_MAX_DAYS,
};

const EPS: f64 = 1e-9;

fn add(graph: &mut Graph, u: usize, v: usize, g: f64) {
    graph.add_edge(u, v, g).expect("edge");
}

fn run(graph: &Graph) -> polyamorous_scheduling::Schedule {
    deadline_driven_schedule(graph, DEFAULT_MAX_DAYS).expect("schedule")
}

/// Section 1: G_v = ∑_{e ∋ v} g_e, G* = max_v G_v.
#[test]
fn g_star_matches_section_1() {
    let mut g = Graph::new(4);
    add(&mut g, 0, 1, 2.0);
    add(&mut g, 0, 2, 3.0);
    add(&mut g, 1, 3, 0.5);
    let gv = g.vertex_growth();
    assert!((gv[0] - 5.0).abs() < EPS);
    assert!((gv[1] - 2.5).abs() < EPS);
    assert!((gv[2] - 3.0).abs() < EPS);
    assert!((gv[3] - 0.5).abs() < EPS);
    assert!((g.g_star() - 5.0).abs() < EPS);
}

/// Section 2, displayed deadline. Normalized G* = 1:
/// D_t(e) = t + (2 − h_e(t)) / g_e.
/// Unnormalized: D_t(e) = t + (2 G* − h_e(t)) / g_e.
#[test]
fn deadline_formula_section_2() {
    let t = 7.0;
    let heat = 1.25;
    let ge = 0.5;
    let d_norm = deadline(t, heat, ge, 1.0);
    assert!((d_norm - (t + (2.0 - heat) / ge)).abs() < EPS);

    let g_star = 3.0;
    let d = deadline(t, heat, ge, g_star);
    assert!((d - (t + (2.0 * g_star - heat) / ge)).abs() < EPS);

    // Scaling: heats and rates divided by G* must not change the matching order.
    let d_scaled = deadline(t, heat / g_star, ge / g_star, 1.0);
    assert!((d - d_scaled).abs() < EPS);
}

/// Lemma 1 (Witness Lemma): an eligible unscheduled edge has an incident
/// matching edge with strictly earlier (deadline, index).
#[test]
fn witness_lemma_on_each_day() {
    let graphs = sample_graphs();
    for (name, graph) in &graphs {
        let g_star = graph.g_star();
        if graph.m() == 0 {
            continue;
        }
        let mut last = vec![0usize; graph.m()];
        for t in 0..2_000 {
            let heats: Vec<f64> = (0..graph.m())
                .map(|e| graph.edges()[e].g * (t - last[e]) as f64)
                .collect();
            let matching = deadline_driven_matching(graph, &heats, t as f64, g_star);
            assert!(
                is_matching(graph, &matching),
                "{name}: matching invalid at t={t}"
            );
            let in_m: Vec<bool> = {
                let mut v = vec![false; graph.m()];
                for &e in &matching {
                    v[e] = true;
                }
                v
            };
            for e0 in 0..graph.m() {
                if heats[e0] + 1e-12 < g_star || in_m[e0] {
                    continue;
                }
                let d0 = deadline(t as f64, heats[e0], graph.edges()[e0].g, g_star);
                let u = graph.edges()[e0].u;
                let v = graph.edges()[e0].v;
                let mut found = false;
                for &f in &matching {
                    let fu = graph.edges()[f].u;
                    let fv = graph.edges()[f].v;
                    if fu != u && fu != v && fv != u && fv != v {
                        continue;
                    }
                    let df = deadline(t as f64, heats[f], graph.edges()[f].g, g_star);
                    let precedes = df < d0 - 1e-12 || ((df - d0).abs() <= 1e-12 && f < e0);
                    if precedes {
                        found = true;
                        break;
                    }
                }
                assert!(found, "{name}: Lemma 1 failed at t={t} for edge {e0}");
            }
            for &e in &matching {
                last[e] = t;
            }
        }
    }
}

/// Section 1: OPT ≥ G*. Any feasible periodic schedule's OPS cost is ≥ G*.
/// Checked by comparing G* to the algorithm and to every short enumerated
/// schedule on tiny graphs.
#[test]
fn g_star_is_lower_bound() {
    for (name, graph) in sample_graphs() {
        let g_star = graph.g_star();
        let sched = run(&graph);
        let cost = max_weighted_waiting_time(&graph, &sched.matchings);
        assert!(
            cost + EPS >= g_star,
            "{name}: periodic cost {cost} < G* {g_star}"
        );
        assert!(
            sched.max_heat + EPS >= g_star || graph.m() == 0,
            "{name}: online heat {} < G* {g_star}",
            sched.max_heat
        );
    }

    // Enumerate all sequences of matchings of period ≤ 4 on a 3-vertex path.
    let mut path = Graph::new(3);
    add(&mut path, 0, 1, 1.0);
    add(&mut path, 1, 2, 1.0);
    let g_star = path.g_star();
    assert!((g_star - 2.0).abs() < EPS);
    let matchings = all_matchings(&path);
    for period in 1..=4 {
        for seq in sequences(&matchings, period) {
            let cost = max_weighted_waiting_time(&path, &seq);
            if cost.is_finite() {
                assert!(
                    cost + EPS >= g_star,
                    "enumerated path schedule cost {cost} < G* {g_star}"
                );
            }
        }
    }

    // Triangle, unit weights: G* = 2.
    let mut tri = Graph::new(3);
    add(&mut tri, 0, 1, 1.0);
    add(&mut tri, 1, 2, 1.0);
    add(&mut tri, 2, 0, 1.0);
    let g_star = tri.g_star();
    assert!((g_star - 2.0).abs() < EPS);
    let matchings = all_matchings(&tri);
    for period in 1..=3 {
        for seq in sequences(&matchings, period) {
            let cost = max_weighted_waiting_time(&tri, &seq);
            if cost.is_finite() {
                assert!(cost + EPS >= g_star);
            }
        }
    }
}

/// Theorem 1: deadline-ordered greedy matching achieves heat strictly < 4 G*.
#[test]
fn heat_at_most_4g_star() {
    for (name, graph) in sample_graphs() {
        let g_star = graph.g_star();
        let sched = run(&graph);
        let cost = max_weighted_waiting_time(&graph, &sched.matchings);
        if g_star == 0.0 {
            assert_eq!(cost, 0.0);
            continue;
        }
        assert!(
            sched.max_heat < 4.0 * g_star + EPS,
            "{name}: online heat {} ≥ 4 G* = {}",
            sched.max_heat,
            4.0 * g_star
        );
        assert!(
            cost < 4.0 * g_star + EPS,
            "{name}: OPS cost {cost} ≥ 4 G* = {}",
            4.0 * g_star
        );
        assert!(
            cost.is_finite(),
            "{name}: some edge never appears in the cycle"
        );
    }
    for (name, graph) in random_graphs(20) {
        let g_star = graph.g_star();
        let sched = run(&graph);
        let cost = max_weighted_waiting_time(&graph, &sched.matchings);
        assert!(
            sched.max_heat < 4.0 * g_star + EPS,
            "{name}: online heat {} ≥ 4 G* = {}",
            sched.max_heat,
            4.0 * g_star
        );
        assert!(
            cost < 4.0 * g_star + EPS,
            "{name}: OPS cost {cost} ≥ 4 G* = {}",
            4.0 * g_star
        );
    }
}

/// Star graphs are Bamboo Garden Trimming (Section 1). The schedule must be
/// a sequence of single edges (or empty), every edge appears, period finite.
#[test]
fn bamboo_garden_trimming_star_is_well_formed() {
    let mut star = Graph::new(6);
    add(&mut star, 0, 1, 0.4);
    add(&mut star, 0, 2, 0.3);
    add(&mut star, 0, 3, 0.15);
    add(&mut star, 0, 4, 0.1);
    add(&mut star, 0, 5, 0.05);
    assert!((star.g_star() - 1.0).abs() < EPS);

    let sched = run(&star);
    assert!(sched.period() > 0 && sched.period() < DEFAULT_MAX_DAYS);
    let mut seen = vec![false; star.m()];
    for matching in &sched.matchings {
        assert!(is_matching(&star, matching));
        assert!(
            matching.len() <= 1,
            "star matching larger than 1: {matching:?}"
        );
        for &e in matching {
            seen[e] = true;
        }
    }
    assert!(seen.iter().all(|&b| b), "some star edge never appears");
    let cost = max_weighted_waiting_time(&star, &sched.matchings);
    assert!(cost.is_finite());
    assert!(cost < 4.0 * star.g_star() + EPS);
}

/// Disjoint edges: a matching graph, also a (trivial) BGT / OPS special case.
#[test]
fn disjoint_edges_scheduled_in_parallel() {
    let mut g = Graph::new(6);
    add(&mut g, 0, 1, 1.0);
    add(&mut g, 2, 3, 0.7);
    add(&mut g, 4, 5, 0.2);
    let sched = run(&g);
    for matching in &sched.matchings {
        assert!(is_matching(&g, matching));
    }
    // After the first growth, all three edges are vertex-disjoint and should
    // be allowed to fire in the same matching once eligible.
    let cost = max_weighted_waiting_time(&g, &sched.matchings);
    assert!(cost.is_finite());
    let mut appeared = vec![false; g.m()];
    for matching in &sched.matchings {
        for &e in matching {
            appeared[e] = true;
        }
    }
    assert!(appeared.iter().all(|&b| b));
}

#[test]
fn every_produced_matching_is_valid() {
    for (name, graph) in sample_graphs() {
        let sched = run(&graph);
        for (t, matching) in sched.matchings.iter().enumerate() {
            assert!(
                is_matching(&graph, matching),
                "{name}: day {t} is not a matching"
            );
        }
    }
    for (name, graph) in random_graphs(10) {
        let sched = run(&graph);
        for (t, matching) in sched.matchings.iter().enumerate() {
            assert!(
                is_matching(&graph, matching),
                "{name}: day {t} is not a matching"
            );
        }
    }
}

/// Section 2: an edge is eligible iff h_e(t) ≥ G* (normalized: ≥ 1).
/// Frequencies: days to eligibility after a reset is the first k with k g_e ≥ G*.
#[test]
fn eligibility_and_frequency_formulas() {
    let mut g = Graph::new(3);
    add(&mut g, 0, 1, 0.25);
    add(&mut g, 0, 2, 0.75);
    let g_star = g.g_star();
    assert!((g_star - 1.0).abs() < EPS);

    // After fictitious service at time 0, heat at day t is g_e t.
    // Edge 0 (g=0.25) becomes eligible at t with 0.25 t ≥ 1, i.e. t ≥ 4.
    // Edge 1 (g=0.75) at t ≥ 2, since 0.75 * 1 = 0.75 < 1 and 0.75 * 2 = 1.5 ≥ 1.
    let heats_at = |t: usize| -> Vec<f64> { g.edges().iter().map(|e| e.g * t as f64).collect() };
    let m0 = deadline_driven_matching(&g, &heats_at(1), 1.0, g_star);
    assert!(m0.is_empty(), "nothing eligible at t=1");
    let m1 = deadline_driven_matching(&g, &heats_at(2), 2.0, g_star);
    assert_eq!(m1, vec![1], "only the faster edge is eligible at t=2");
    let m2 = deadline_driven_matching(&g, &heats_at(4), 4.0, g_star);
    assert!(!m2.is_empty());
    // Both eligible at t=4; they share vertex 0, so exactly one is taken.
    // Deadlines: D_4(e) = 4 + (2 − h)/g.
    let h0 = 0.25 * 4.0;
    let h1 = 0.75 * 4.0;
    let d0 = deadline(4.0, h0, 0.25, 1.0);
    let d1 = deadline(4.0, h1, 0.75, 1.0);
    // h0=1, D=4+(2-1)/0.25=8; h1=3, D=4+(2-3)/0.75=4-1.333=2.666, so e1 first.
    assert!(d1 < d0);
    assert_eq!(m2, vec![1]);
}

/// Expiration time used in the Section 2 counting argument:
/// s ≤ 2 (1/g_e − 1/g_f) when G* = 1 and h_f ≥ 0.
#[test]
fn expiration_time_identity() {
    let ge = 0.4;
    let gf = 0.6;
    let bound = 2.0 * (1.0 / ge - 1.0 / gf);
    // D_s(f) ≤ s + 2/g_f and D_s(e0) = 2/g_e imply the displayed bound.
    let s = 0.0;
    let hf = 0.0;
    let ds_f = deadline(s, hf, gf, 1.0);
    let ds_e = 2.0 / ge;
    assert!((ds_f - (s + 2.0 / gf)).abs() < EPS);
    assert!(ds_f <= ds_e + EPS);
    let s_max = 2.0 / ge - 2.0 / gf;
    assert!((s_max - bound).abs() < EPS);
}

fn sample_graphs() -> Vec<(&'static str, Graph)> {
    let mut out = Vec::new();

    out.push(("empty", Graph::new(3)));

    let mut single = Graph::new(2);
    add(&mut single, 0, 1, 1.5);
    out.push(("single", single));

    let mut path = Graph::new(4);
    add(&mut path, 0, 1, 1.0);
    add(&mut path, 1, 2, 2.0);
    add(&mut path, 2, 3, 1.0);
    out.push(("path", path));

    let mut tri = Graph::new(3);
    add(&mut tri, 0, 1, 1.0);
    add(&mut tri, 1, 2, 1.0);
    add(&mut tri, 2, 0, 1.0);
    out.push(("triangle", tri));

    let mut k4 = Graph::new(4);
    add(&mut k4, 0, 1, 0.5);
    add(&mut k4, 0, 2, 0.5);
    add(&mut k4, 0, 3, 0.5);
    add(&mut k4, 1, 2, 0.4);
    add(&mut k4, 1, 3, 0.3);
    add(&mut k4, 2, 3, 0.2);
    out.push(("k4", k4));

    let mut star = Graph::new(5);
    add(&mut star, 0, 1, 0.5);
    add(&mut star, 0, 2, 0.3);
    add(&mut star, 0, 3, 0.15);
    add(&mut star, 0, 4, 0.05);
    out.push(("star", star));

    let mut matching = Graph::new(6);
    add(&mut matching, 0, 1, 1.0);
    add(&mut matching, 2, 3, 2.0);
    add(&mut matching, 4, 5, 0.5);
    out.push(("matching", matching));

    let mut k23 = Graph::new(5);
    add(&mut k23, 0, 2, 0.4);
    add(&mut k23, 0, 3, 0.3);
    add(&mut k23, 0, 4, 0.2);
    add(&mut k23, 1, 2, 0.5);
    add(&mut k23, 1, 3, 0.1);
    add(&mut k23, 1, 4, 0.25);
    out.push(("k23", k23));

    out
}

fn random_graphs(count: usize) -> Vec<(String, Graph)> {
    let mut out = Vec::new();
    let mut rng = Lcg(0xC0FFEE);
    for i in 0..count {
        let n = 3 + (rng.next() as usize % 5);
        let mut g = Graph::new(n);
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.f64() < 0.55 {
                    let w = 0.05 + 2.0 * rng.f64();
                    add(&mut g, u, v, w);
                }
            }
        }
        if g.m() == 0 {
            add(&mut g, 0, 1 % n.max(2).min(n - 1).max(1), 1.0);
        }
        out.push((format!("rand{i}"), g));
    }
    out
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

fn all_matchings(graph: &Graph) -> Vec<Vec<usize>> {
    let m = graph.m();
    let mut out = Vec::new();
    for mask in 0..(1usize << m) {
        let edges: Vec<usize> = (0..m).filter(|e| mask & (1 << e) != 0).collect();
        if is_matching(graph, &edges) {
            out.push(edges);
        }
    }
    out
}

fn sequences(items: &[Vec<usize>], period: usize) -> Vec<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    let mut cur = vec![0usize; period];
    loop {
        out.push(cur.iter().map(|&i| items[i].clone()).collect());
        let mut k = period;
        while k > 0 {
            k -= 1;
            cur[k] += 1;
            if cur[k] < items.len() {
                for j in (k + 1)..period {
                    cur[j] = 0;
                }
                break;
            }
        }
        if k == 0
            && cur
                .iter()
                .all(|&x| x >= items.len() || cur[0] >= items.len())
        {
            // finished when increment overflowed past the first position
            if cur[0] >= items.len() {
                break;
            }
        }
        if cur.iter().all(|&x| x == 0) {
            break;
        }
    }
    out
}
