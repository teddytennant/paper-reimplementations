//! Lemma 5's guarantees on `Zorro`, the base cases of Algorithm 2, and
//! reproducibility.

mod common;

use common::check_bispanner;
use temporal_spanner::reach::biclique_graph;
use temporal_spanner::{
    bounded_span, clique_spanner, default_f, random_biclique, random_clique, rec_span,
    rec_span_with, zorro, TemporalBiclique,
};

#[test]
fn zorro_returns_the_promised_set_sizes() {
    for n in [2usize, 3, 5, 9, 16, 30] {
        for k in 1..n {
            for seed in 0..3u64 {
                let g = random_biclique(n, 25, seed + 3 * n as u64 + 5 * k as u64);
                let out = zorro(&g, &g.all_sources(), &g.all_targets(), k);
                assert_eq!(out.s1.len(), k, "|S_1| should be k");
                assert_eq!(out.t2.len(), n - k, "|T_2| should be n-k");
                let mut s1 = out.s1.clone();
                s1.sort_unstable();
                s1.dedup();
                assert_eq!(s1.len(), k, "S_1 has repeats");
                let mut t2 = out.t2.clone();
                t2.sort_unstable();
                t2.dedup();
                assert_eq!(t2.len(), n - k, "T_2 has repeats");
            }
        }
    }
}

#[test]
fn zorro_uses_at_most_n_edges_forming_two_stars() {
    for n in [2usize, 4, 8, 17, 40] {
        for k in 1..n {
            let g = random_biclique(n, 25, 11 * n as u64 + k as u64);
            let out = zorro(&g, &g.all_sources(), &g.all_targets(), k);
            assert!(out.z.len() <= n, "|Z| = {} exceeds n = {n}", out.z.len());
            for &(s, t) in &out.z {
                let in_first = t == out.t_star && out.s1.contains(&s);
                let in_second = s == out.s_star && out.t2.contains(&t);
                assert!(in_first || in_second, "edge ({s},{t}) is in neither star");
            }
            assert!(out.s1.contains(&out.s_star), "s* must lie in S_1");
            assert!(!out.t2.contains(&out.t_star), "t* must not lie in T_2");
        }
    }
}

#[test]
fn the_zorro_path_is_temporal() {
    // lambda(s,t*) <= lambda(s*,t*) <= lambda(s*,t), the two inequalities of the
    // proof of Lemma 5, checked one at a time.
    for n in [2usize, 3, 6, 12, 25] {
        for k in 1..n {
            for max_label in [2u32, 5, 100] {
                let g = random_biclique(
                    n,
                    max_label,
                    7 * n as u64 + 13 * k as u64 + max_label as u64,
                );
                let out = zorro(&g, &g.all_sources(), &g.all_targets(), k);
                let mid = g.label(out.s_star, out.t_star);
                for &s in &out.s1 {
                    assert!(
                        g.label(s, out.t_star) <= mid,
                        "first hop: lambda({s},t*) = {} > lambda(s*,t*) = {mid}",
                        g.label(s, out.t_star)
                    );
                }
                for &t in &out.t2 {
                    assert!(
                        mid <= g.label(out.s_star, t),
                        "third hop: lambda(s*,t*) = {mid} > lambda(s*,{t}) = {}",
                        g.label(out.s_star, t)
                    );
                }
                for &s in &out.s1 {
                    for &t in &out.t2 {
                        let a = g.label(s, out.t_star);
                        let b = g.label(out.s_star, t);
                        assert!(
                            a <= mid && mid <= b,
                            "path ({s}, t*, s*, {t}) is not temporal"
                        );
                        assert!(out.z.contains(&(s, out.t_star)));
                        assert!(out.z.contains(&(out.s_star, t)));
                    }
                }
            }
        }
    }
}

#[test]
fn zorro_covers_s1_by_t2_according_to_the_independent_checker() {
    for n in [3usize, 7, 14] {
        for k in 1..n {
            let g = random_biclique(n, 30, 23 * n as u64 + k as u64);
            let out = zorro(&g, &g.all_sources(), &g.all_targets(), k);
            let graph = biclique_graph(n, n, &out.z, |s, t| g.label(s, t));
            for &s in &out.s1 {
                for &t in &out.t2 {
                    let d = graph.hop_distances_from(s, 2 * n)[n + t]
                        .unwrap_or_else(|| panic!("Z does not connect {s} to {t}"));
                    assert!(d <= 3, "Z connects {s} to {t} only in {d} hops");
                }
            }
        }
    }
}

#[test]
fn f_stays_in_range() {
    for n in 3..3000usize {
        let k = default_f(n);
        assert!((2..n).contains(&k), "f({n}) = {k} is outside 2..={}", n - 1);
    }
    assert_eq!(default_f(3), 2);
    assert_eq!(default_f(4), 2);
}

#[test]
fn base_case_returns_every_edge() {
    for n in [1usize, 2] {
        let g = random_biclique(n, 9, n as u64);
        let e = rec_span(&g);
        assert_eq!(e.len(), n * n, "n={n} should return the whole edge set");
        check_bispanner(&g, &e.to_vec());
    }
}

#[test]
fn n_three_is_where_recursion_first_fires() {
    use std::cell::RefCell;
    let seen: RefCell<Vec<usize>> = RefCell::new(Vec::new());

    // n <= 2 is the base case of Algorithm 2 and never asks for k.
    let g2 = random_biclique(2, 9, 4242);
    let e2 = rec_span_with(&g2, |n| {
        seen.borrow_mut().push(n);
        default_f(n)
    });
    assert!(seen.borrow().is_empty(), "n=2 should not call f");
    assert_eq!(e2.len(), 4);

    // f(3) = 2, so n = 3 calls Zorro twice and recurses on sizes 2, 2 and 1,
    // which are all base cases.
    let g3 = random_biclique(3, 9, 4242);
    let e3 = rec_span_with(&g3, |n| {
        seen.borrow_mut().push(n);
        default_f(n)
    });
    assert_eq!(
        *seen.borrow(),
        vec![3],
        "n=3 should call f once and then hit base cases"
    );
    assert!(e3.len() <= 9);
    check_bispanner(&g3, &e3.to_vec());
}

#[test]
fn all_labels_equal_still_spans() {
    for n in [1usize, 2, 3, 6, 15, 31] {
        let g = TemporalBiclique::from_labels(n, n, vec![4; n * n]);
        check_bispanner(&g, &rec_span(&g).to_vec());
        check_bispanner(&g, &bounded_span(&g, 4).to_vec());
        // Lifetime 1 has its own trivial branch: a source star plus a target star.
        let flat = TemporalBiclique::from_labels(n, n, vec![1; n * n]);
        let e = bounded_span(&flat, 1);
        assert_eq!(
            e.len(),
            2 * n - 1,
            "n={n}: lifetime 1 should cost 2n-1 edges"
        );
        check_bispanner(&flat, &e.to_vec());
    }
}

#[test]
fn the_same_seed_gives_the_same_spanner() {
    for n in [3usize, 9, 40] {
        for seed in [0u64, 17, 99] {
            let a = random_clique(n, 60, seed);
            let b = random_clique(n, 60, seed);
            assert_eq!(a.edges(), b.edges());
            assert_eq!(clique_spanner(&a), clique_spanner(&b));
            assert_eq!(clique_spanner(&a), clique_spanner(&a));

            let ga = random_biclique(n, 5, seed);
            let gb = random_biclique(n, 5, seed);
            assert_eq!(rec_span(&ga).to_vec(), rec_span(&gb).to_vec());
            assert_eq!(bounded_span(&ga, 5).to_vec(), bounded_span(&gb, 5).to_vec());
        }
    }
}

#[test]
fn different_seeds_give_different_instances() {
    let a = random_clique(30, 500, 1);
    let b = random_clique(30, 500, 2);
    let differs = (0..30).any(|u| (0..30).any(|v| u != v && a.label(u, v) != b.label(u, v)));
    assert!(differs, "two seeds produced the same clique");
}
