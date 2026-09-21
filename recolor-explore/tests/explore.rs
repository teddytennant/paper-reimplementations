//! Exploration tests: every adversary branch visits all vertices, returns, stops.

use recolor_explore::{
    explore_ac4, explore_ac4_all_starts, explore_atc3, explore_atc3_all_starts, Graph,
};

fn atc3_ok(g: &Graph, label: &str) -> usize {
    let n = g.n();
    match explore_atc3_all_starts(g, 50 * n.max(1)) {
        Ok(steps) => steps,
        Err(e) => panic!("ATC3 failed on {label} ({g}): {e}"),
    }
}

fn ac4_ok(g: &Graph, label: &str) -> usize {
    let n = g.n();
    match explore_ac4_all_starts(g, 50 * n.max(1)) {
        Ok(steps) => steps,
        Err(e) => panic!("AC4 failed on {label} ({g}): {e}"),
    }
}

fn for_each_labeled_tree(n: usize, mut f: impl FnMut(Graph)) {
    if n == 0 {
        return;
    }
    if n == 1 {
        f(Graph::new(1));
        return;
    }
    if n == 2 {
        f(Graph::path(2));
        return;
    }
    let len = n - 2;
    let mut code = vec![0usize; len];
    loop {
        f(Graph::from_prufer(&code));
        let mut i = 0;
        while i < len {
            code[i] += 1;
            if code[i] < n {
                break;
            }
            code[i] = 0;
            i += 1;
        }
        if i == len {
            break;
        }
    }
}

#[test]
fn atc3_isolated_vertex() {
    let g = Graph::new(1);
    let steps = explore_atc3(&g, 0, 50).expect("ATC3 isolated");
    assert_eq!(steps, 1, "isolated vertex stops after one recoloring");
}

#[test]
fn ac4_isolated_vertex() {
    let g = Graph::new(1);
    let steps = explore_ac4(&g, 0, 50).expect("AC4 isolated");
    assert_eq!(steps, 1, "isolated vertex stops after one recoloring");
}

#[test]
fn atc3_named_trees() {
    for (g, label) in [
        (Graph::path(2), "P2"),
        (Graph::path(3), "P3"),
        (Graph::path(4), "P4"),
        (Graph::star(2), "K1,1"),
        (Graph::star(4), "K1,3"),
        (Graph::star(7), "K1,6"),
        (Graph::binaryish5(), "binaryish5"),
        (Graph::balanced_binary7(), "balanced_binary7"),
    ] {
        let n = g.n();
        let steps = atc3_ok(&g, label);
        assert!(
            steps <= 20 * n,
            "ATC3 {label}: {steps} moves > 20n={}",
            20 * n
        );
    }
}

#[test]
fn atc3_every_tree_n_le_7() {
    let mut max_ratio_steps = 0usize;
    let mut max_n = 1usize;
    for n in 1..=7 {
        for_each_labeled_tree(n, |g| {
            let steps = atc3_ok(&g, &format!("tree n={n} {}", g));
            assert!(steps <= 20 * n, "ATC3 tree {}: {steps} > 20n={}", g, 20 * n);
            if steps > max_ratio_steps {
                max_ratio_steps = steps;
                max_n = n;
            }
        });
    }
    let _ = (max_ratio_steps, max_n);
}

#[test]
fn atc3_every_cycle_3_to_10() {
    for n in 3..=10 {
        let g = Graph::cycle(n);
        let steps = atc3_ok(&g, &format!("C{n}"));
        assert!(steps <= 20 * n, "ATC3 C{n}: {steps} moves > 20n={}", 20 * n);
    }
}

#[test]
fn atc3_move_count_on_named_and_cycles() {
    // Dedicated O(n) bound check (also asserted in the exploration tests).
    for n in 1..=7 {
        for_each_labeled_tree(n, |g| {
            let steps = atc3_ok(&g, "bound");
            assert!(steps <= 20 * n);
        });
    }
    for n in 3..=10 {
        let steps = atc3_ok(&Graph::cycle(n), "bound cycle");
        assert!(steps <= 20 * n);
    }
}

#[test]
fn ac4_single_edges() {
    let g = Graph::path(2);
    let steps = ac4_ok(&g, "P2");
    assert!(steps <= 30 * g.n());
}

#[test]
fn ac4_cycles_c3_to_c8() {
    for n in 3..=8 {
        let g = Graph::cycle(n);
        let steps = ac4_ok(&g, &format!("C{n}"));
        assert!(steps <= 30 * n, "AC4 C{n}: {steps} moves > 30n={}", 30 * n);
    }
}

#[test]
fn ac4_complete_bipartite() {
    for (p, q) in [(2, 2), (2, 3), (3, 3)] {
        let g = Graph::complete_bipartite(p, q);
        let steps = ac4_ok(&g, &format!("K{p},{q}"));
        assert!(
            steps <= 30 * g.n(),
            "AC4 K{p},{q}: {steps} > 30n={}",
            30 * g.n()
        );
    }
}

#[test]
fn ac4_cactus_two_cycles_share_vertex() {
    let g = Graph::bowtie();
    let steps = ac4_ok(&g, "bowtie");
    assert!(steps <= 30 * g.n());
}

#[test]
fn ac4_two_cycles_joined_by_bridge() {
    let g = Graph::two_cycles_bridge();
    let steps = ac4_ok(&g, "two-cycles-bridge");
    assert!(steps <= 30 * g.n());
}

#[test]
fn ac4_stars() {
    for q in 1..=6 {
        let g = Graph::star(q + 1);
        let steps = ac4_ok(&g, &format!("K1,{q}"));
        assert!(
            steps <= 30 * g.n(),
            "AC4 K1,{q}: {steps} > 30n={}",
            30 * g.n()
        );
    }
}

#[test]
fn ac4_move_count_on_required_graphs() {
    let graphs: Vec<(Graph, String)> = {
        let mut v = vec![
            (Graph::path(2), "P2".into()),
            (Graph::bowtie(), "bowtie".into()),
            (Graph::two_cycles_bridge(), "bridge".into()),
        ];
        for n in 3..=8 {
            v.push((Graph::cycle(n), format!("C{n}")));
        }
        for (p, q) in [(2, 2), (2, 3), (3, 3)] {
            v.push((Graph::complete_bipartite(p, q), format!("K{p},{q}")));
        }
        for q in 1..=6 {
            v.push((Graph::star(q + 1), format!("K1,{q}")));
        }
        v
    };
    for (g, label) in graphs {
        let steps = ac4_ok(&g, &label);
        assert!(
            steps <= 30 * g.n(),
            "AC4 {label}: {steps} > 30n={}",
            30 * g.n()
        );
    }
}
