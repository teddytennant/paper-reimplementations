//! Tests that exercise the claims of Demin, arXiv:2609.14433.

use prob_ref_f4::{
    algorithm1, all_rows_reduce_to_zero, certify_or_fallback, certified_ref, deterministic_ref,
    las_vegas_ref, success_probability, Fp, Matrix, RowEchelon,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rng() -> StdRng {
    StdRng::seed_from_u64(260914433)
}

fn full_rank_triangular(n: usize, field: Fp) -> Matrix {
    let mut m = Matrix::identity(n, field);
    // Strictly upper triangular fill; diagonal stays 1, so rank n.
    for i in 0..n {
        for j in (i + 1)..n {
            m.set(i, j, field.reduce((i + j + 1) as i64));
        }
    }
    m
}

fn rank_deficient(n: usize, rank: usize, field: Fp, rng: &mut StdRng) -> Matrix {
    assert!(rank <= n);
    let mut m = Matrix::zeros(n, n, field);
    for i in 0..rank {
        for j in 0..n {
            m.set(i, j, rng.gen_range(0..field.modulus()));
        }
        if m.row(i).iter().all(|&x| x == 0) {
            m.set(i, i % n, 1);
        }
    }
    for i in rank..n {
        // Copy a linear combination of the first `rank` rows.
        for k in 0..rank {
            let c = rng.gen_range(0..field.modulus());
            for j in 0..n {
                let term = field.mul(c, m.get(k, j));
                m.set(i, j, field.add(m.get(i, j), term));
            }
        }
    }
    m
}

#[test]
fn probability_formula_matches_mc_gf2() {
    // Corollary 3 is exact for Algorithm 1 when ell = N and A has full row rank.
    let field = Fp::GF2;
    let r = 4usize;
    let s = 1usize;
    let closed = success_probability(2, r, s);
    // prod_{i=1}^{4} (1 - 1/2^i) = (1/2)(3/4)(7/8)(15/16)
    let expected = 0.5 * 0.75 * 0.875 * 0.9375;
    assert!((closed - expected).abs() < 1e-12);

    let a = Matrix::identity(r, field);
    let trials = 8000usize;
    let mut hits = 0usize;
    let mut rng = rng();
    for _ in 0..trials {
        let l = algorithm1(&a, 0, r, s, &mut rng);
        if l.rank() == r {
            hits += 1;
        }
    }
    let mc = hits as f64 / trials as f64;
    assert!(
        (mc - closed).abs() < 0.04,
        "MC {mc} vs closed {closed} (r={r}, s={s}, q=2)"
    );
}

#[test]
fn probability_formula_matches_mc_s2() {
    let field = Fp::GF2;
    let r = 3usize;
    let s = 2usize;
    let closed = success_probability(2, r, s);
    // prod_{i=2}^{4} (1-1/2^i) = (3/4)(7/8)(15/16)
    let expected = 0.75 * 0.875 * 0.9375;
    assert!((closed - expected).abs() < 1e-12);

    let a = Matrix::identity(r, field);
    let trials = 6000usize;
    let mut hits = 0usize;
    let mut rng = rng();
    for _ in 0..trials {
        let l = algorithm1(&a, 0, r, s, &mut rng);
        if l.rank() == r {
            hits += 1;
        }
    }
    let mc = hits as f64 / trials as f64;
    assert!(
        (mc - closed).abs() < 0.04,
        "MC {mc} vs closed {closed} (r={r}, s={s}, q=2)"
    );
}

#[test]
fn probability_formula_matches_mc_gf3() {
    let field = Fp::new(3);
    let r = 3usize;
    let s = 1usize;
    let closed = success_probability(3, r, s);
    let a = Matrix::identity(r, field);
    let trials = 5000usize;
    let mut hits = 0usize;
    let mut rng = rng();
    for _ in 0..trials {
        let l = algorithm1(&a, 0, r, s, &mut rng);
        if l.rank() == r {
            hits += 1;
        }
    }
    let mc = hits as f64 / trials as f64;
    assert!(
        (mc - closed).abs() < 0.04,
        "MC {mc} vs closed {closed} (q=3)"
    );
}

#[test]
fn known_full_rank_square_las_vegas() {
    for field in [Fp::GF2, Fp::P32003, Fp::P65521] {
        let a = full_rank_triangular(6, field);
        let mut rng = rng();
        let l = certified_ref(&a, 2, &mut rng);
        assert_eq!(l.rank(), 6, "field {}", field.modulus());
        assert!(l.is_row_echelon());
        assert!(a.rowspace_eq(&l.to_matrix()));
        assert!(all_rows_reduce_to_zero(&l, &a));
    }
}

#[test]
fn rank_deficient_after_certification() {
    let mut rng = rng();
    for field in [Fp::GF2, Fp::P32003] {
        let a = rank_deficient(8, 3, field, &mut rng);
        let det = deterministic_ref(&a);
        assert_eq!(det.rank(), 3, "setup rank");
        let l = certified_ref(&a, 2, &mut rng);
        assert_eq!(l.rank(), 3, "certified rank, p={}", field.modulus());
        assert!(a.rowspace_eq(&l.to_matrix()));
    }
}

#[test]
fn output_rows_in_input_rowspace_before_certification() {
    let mut rng = rng();
    let field = Fp::P32003;
    let a = Matrix::random(10, 7, field, &mut rng);
    let l = algorithm1(&a, 0, 4, 1, &mut rng);
    // Proposition 1: rowspace(L) subset rowspace(A) always.
    assert!(a.rowspace_contains(&l.to_matrix()));
    assert!(l.is_row_echelon());
}

#[test]
fn certification_detects_truncated_l() {
    let field = Fp::P32003;
    let a = Matrix::identity(5, field);
    // Truncated identity: rank 3, does not span the original rows.
    let truncated = RowEchelon::from_prefix(
        field,
        &[
            vec![1, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0],
            vec![0, 0, 1, 0, 0],
        ],
    );
    assert_eq!(truncated.rank(), 3);
    assert!(
        !all_rows_reduce_to_zero(&truncated, &a),
        "truncated L must fail the rowspace check"
    );
    let certified = certify_or_fallback(&truncated, &a);
    assert_eq!(certified.rank(), 5);
    assert!(a.rowspace_eq(&certified.to_matrix()));
}

#[test]
fn deterministic_and_certified_agree_on_rank() {
    let mut rng = rng();
    let shapes = [(0, 0), (1, 1), (3, 8), (8, 3), (7, 7), (5, 5)];
    for field in [Fp::GF2, Fp::P32003] {
        for &(r, c) in &shapes {
            let a = if r == 0 && c == 0 {
                Matrix::zeros(0, 0, field)
            } else {
                Matrix::random(r, c, field, &mut rng)
            };
            let d = deterministic_ref(&a);
            let p = if r == 0 {
                RowEchelon::empty(field, c)
            } else {
                certified_ref(&a, 1, &mut rng)
            };
            assert_eq!(
                d.rank(),
                p.rank(),
                "rank mismatch p={} {}x{}",
                field.modulus(),
                r,
                c
            );
            assert!(d.to_matrix().rowspace_eq(&p.to_matrix()));
        }
    }
}

#[test]
fn edge_empty() {
    let f = Fp::GF2;
    let a = Matrix::zeros(0, 0, f);
    assert_eq!(deterministic_ref(&a).rank(), 0);
    let mut rng = rng();
    let l = certified_ref(&a, 1, &mut rng);
    assert_eq!(l.rank(), 0);

    let wide = Matrix::zeros(0, 4, f);
    assert_eq!(deterministic_ref(&wide).rank(), 0);
    let tall = Matrix::zeros(4, 0, f);
    assert_eq!(deterministic_ref(&tall).rank(), 0);
    let l2 = las_vegas_ref(&tall, 0, 2, 1, &mut rng);
    assert_eq!(l2.rank(), 0);
}

#[test]
fn edge_one_by_one() {
    let mut rng = rng();
    for field in [Fp::GF2, Fp::P32003] {
        let z = Matrix::from_i64_rows(field, &[&[0]]);
        assert_eq!(certified_ref(&z, 1, &mut rng).rank(), 0);
        let o = Matrix::from_i64_rows(field, &[&[1]]);
        assert_eq!(certified_ref(&o, 1, &mut rng).rank(), 1);
        let k = Matrix::from_i64_rows(field, &[&[field.reduce(5)]]);
        let rk = if field.reduce(5) == 0 { 0 } else { 1 };
        assert_eq!(certified_ref(&k, 1, &mut rng).rank(), rk);
    }
}

#[test]
fn edge_already_echelon() {
    let field = Fp::P32003;
    let a = Matrix::identity(5, field);
    let mut rng = rng();
    // R = n: Algorithm 1 returns A1 and never samples A2.
    let l = algorithm1(&a, 5, 1, 1, &mut rng);
    assert_eq!(l.rank(), 5);
    assert!(a.rowspace_eq(&l.to_matrix()));
}

#[test]
fn edge_more_columns_than_rows() {
    let field = Fp::P32003;
    let a = Matrix::from_i64_rows(
        field,
        &[
            &[1, 2, 3, 4, 5, 6],
            &[0, 1, 1, 1, 1, 1],
            &[0, 0, 1, 0, 1, 2],
        ],
    );
    let mut rng = rng();
    let d = deterministic_ref(&a);
    let p = las_vegas_ref(&a, 0, 3, 2, &mut rng);
    assert_eq!(d.rank(), 3);
    assert_eq!(p.rank(), 3);
    assert!(a.rowspace_eq(&p.to_matrix()));
}

#[test]
fn already_echelon_prefix_is_kept() {
    let field = Fp::new(7);
    // A1 already REF of rank 2, A2 has one extra independent row.
    let a = Matrix::from_i64_rows(
        field,
        &[
            &[1, 2, 3, 4],
            &[0, 1, 1, 1],
            &[0, 0, 1, 5],
            &[1, 3, 5, 3],
        ],
    );
    let mut rng = rng();
    let l = las_vegas_ref(&a, 2, 2, 1, &mut rng);
    assert_eq!(l.rank(), deterministic_ref(&a).rank());
    assert!(l.pivots().len() >= 2);
    assert_eq!(l.pivots()[0], 0);
    assert_eq!(l.pivots()[1], 1);
}
