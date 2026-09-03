//! The point of the repo: what comes back really is a temporal 3-spanner.

mod common;

use common::{check_bispanner, check_spanner};
use temporal_spanner::{
    bounded_span, clique_spanner, clique_spanner_bounded, random_biclique, random_clique, rec_span,
};

#[test]
fn recspan_gives_a_3_bispanner() {
    for n in [3usize, 4, 5, 7, 11, 16, 25, 40] {
        for seed in 0..6u64 {
            let g = random_biclique(n, 50, seed * 1000 + n as u64);
            let e = rec_span(&g).to_vec();
            check_bispanner(&g, &e);
        }
    }
}

#[test]
fn recspan_gives_a_3_bispanner_with_few_distinct_labels() {
    // Heavy ties are where a sloppy tie break in Zorro would show up.
    for n in [3usize, 6, 12, 24] {
        for max_label in [1u32, 2, 3, 5] {
            for seed in 0..5u64 {
                let g = random_biclique(n, max_label, seed + 7 * n as u64 + max_label as u64);
                let e = rec_span(&g).to_vec();
                check_bispanner(&g, &e);
            }
        }
    }
}

#[test]
fn clique_spanner_is_a_3_spanner() {
    for n in [2usize, 3, 4, 5, 8, 13, 20, 35] {
        for seed in 0..6u64 {
            let g = random_clique(n, 40, seed * 31 + n as u64);
            let e = clique_spanner(&g);
            check_spanner(&g, &e);
        }
    }
}

#[test]
fn bounded_lifetime_gives_a_3_bispanner() {
    for n in [1usize, 2, 3, 5, 9, 17, 30] {
        for l in [1u32, 2, 3, 4, 8] {
            for seed in 0..4u64 {
                let g = random_biclique(n, l, seed + 13 * n as u64 + 101 * l as u64);
                let e = bounded_span(&g, l).to_vec();
                check_bispanner(&g, &e);
            }
        }
    }
}

#[test]
fn bounded_lifetime_clique_spanner_is_a_3_spanner() {
    for n in [2usize, 3, 4, 6, 10, 18, 30] {
        for l in [2u32, 3, 5] {
            for seed in 0..4u64 {
                let g = random_clique(n, l, seed + 5 * n as u64 + 17 * l as u64);
                let e = clique_spanner_bounded(&g, l);
                check_spanner(&g, &e);
            }
        }
    }
}

#[test]
fn adversarial_label_patterns_still_span() {
    // Sorted, reverse sorted, and constant-per-source labellings, none of which
    // look like the random case.
    let n = 20;
    let patterns: Vec<Box<dyn Fn(usize, usize) -> u32>> = vec![
        Box::new(|s, t| (s + t) as u32 + 1),
        Box::new(|s, t| (n * n) as u32 - (s * n + t) as u32),
        Box::new(|s, _t| s as u32 + 1),
        Box::new(|_s, t| t as u32 + 1),
        Box::new(|s, t| ((s * 7 + t * 13) % 5) as u32 + 1),
    ];
    for p in &patterns {
        let labels: Vec<u32> = (0..n)
            .flat_map(|s| (0..n).map(move |t| (s, t)))
            .map(|(s, t)| p(s, t))
            .collect();
        let g = temporal_spanner::TemporalBiclique::from_labels(n, n, labels);
        let e = rec_span(&g).to_vec();
        check_bispanner(&g, &e);
        let l = g.lifetime();
        let eb = bounded_span(&g, l).to_vec();
        check_bispanner(&g, &eb);
    }
}
