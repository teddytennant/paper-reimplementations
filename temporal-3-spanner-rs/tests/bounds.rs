//! The size bounds: Theorem 4 for the general case, Theorem 12 for bounded
//! lifetime.

use temporal_spanner::{
    bounded_size_bound, bounded_span, clique_spanner, random_biclique, random_clique, rec_span,
    size_bound,
};

#[test]
fn bispanner_meets_the_theorem_4_bound() {
    for n in [1usize, 2, 3, 4, 7, 13, 25, 50, 80, 130, 200] {
        for seed in 0..3u64 {
            let g = random_biclique(n, 1000, seed * 97 + n as u64);
            let size = rec_span(&g).len();
            let bound = size_bound(n);
            assert!(
                (size as f64) <= bound,
                "n={n} seed={seed}: {size} edges exceeds n*e^(2 sqrt ln n) = {bound:.1}"
            );
        }
    }
}

#[test]
fn clique_spanner_meets_the_theorem_1_bound() {
    for n in [2usize, 3, 5, 10, 30, 75, 150] {
        for seed in 0..3u64 {
            let g = random_clique(n, 1000, seed * 11 + n as u64);
            let size = clique_spanner(&g).len();
            assert!(
                (size as f64) <= size_bound(n),
                "n={n} seed={seed}: {size} edges exceeds {:.1}",
                size_bound(n)
            );
        }
    }
}

#[test]
fn spanner_is_well_under_quadratic() {
    // n^(1+2/sqrt(ln n)) is only asymptotically sublinear in n^2, so pin the
    // claim to sizes where the gap is real and check the ratio keeps shrinking.
    let mut prev_ratio = f64::INFINITY;
    for n in [50usize, 100, 200, 400] {
        let g = random_clique(n, 5000, 1234 + n as u64);
        let size = clique_spanner(&g).len();
        let trivial = n * (n - 1) / 2;
        assert!(
            size < trivial / 2,
            "n={n}: {size} edges is not well under {trivial}"
        );
        let ratio = size as f64 / trivial as f64;
        assert!(
            ratio < prev_ratio,
            "n={n}: density {ratio:.4} did not drop below {prev_ratio:.4}"
        );
        prev_ratio = ratio;
    }
}

#[test]
fn bounded_lifetime_meets_the_theorem_12_bound() {
    for n in [1usize, 2, 3, 6, 12, 25, 60, 100] {
        for l in [1u32, 2, 3, 4, 7, 12] {
            for seed in 0..3u64 {
                let g = random_biclique(n, l, seed + 3 * n as u64 + 29 * l as u64);
                let size = bounded_span(&g, l).len();
                let bound = bounded_size_bound(n, l);
                assert!(
                    size <= bound,
                    "n={n} L={l} seed={seed}: {size} edges exceeds 12*L*n = {bound}"
                );
            }
        }
    }
}

#[test]
fn bounded_lifetime_beats_the_general_construction_for_small_l() {
    // Both the proved bounds and the sizes actually produced.
    for n in [60usize, 120, 200] {
        for l in [2u32, 3, 4] {
            assert!(
                (bounded_size_bound(n, l) as f64) < size_bound(n),
                "n={n} L={l}: 12*L*n = {} is not below {:.1}",
                bounded_size_bound(n, l),
                size_bound(n)
            );
            for seed in 0..2u64 {
                let g = random_biclique(n, l, seed + 7 * n as u64 + 41 * l as u64);
                let small = bounded_span(&g, l).len();
                let general = rec_span(&g).len();
                assert!(
                    small < general,
                    "n={n} L={l} seed={seed}: bounded lifetime gave {small}, RecSpan gave {general}"
                );
            }
        }
    }
}
