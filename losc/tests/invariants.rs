use losc::{
    decode, dilution_matrix, measure, sample_pooling, PoolingDesign, PoolingMatrix,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn approx_eq(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn dilution_diagonal_matches_eq1() {
    let phi = PoolingMatrix::from_rows(&[
        vec![1, 1, 0, 1],
        vec![0, 0, 0, 0],
        vec![1, 0, 0, 0],
        vec![1, 1, 1, 1],
    ])
    .unwrap();
    let d = dilution_matrix(&phi);
    assert_eq!(d.len(), 4);
    let sums = [3.0, 0.0, 1.0, 4.0];
    for i in 0..4 {
        if sums[i] == 0.0 {
            assert_eq!(d[i], 0.0);
        } else {
            approx_eq(d[i] * sums[i], 1.0);
            approx_eq(d[i], 1.0 / sums[i]);
        }
    }
}

#[test]
fn pool_viral_load_is_mean_of_members() {
    let phi = PoolingMatrix::from_rows(&[
        vec![1, 1, 0, 0],
        vec![0, 1, 1, 1],
        vec![0, 0, 0, 0],
        vec![1, 0, 0, 1],
    ])
    .unwrap();
    let x = vec![2.0, 4.0, 6.0, 8.0];
    let (y, _) = measure(&phi, &x, 1.0).unwrap();
    approx_eq(y[0], (2.0 + 4.0) / 2.0);
    approx_eq(y[1], (4.0 + 6.0 + 8.0) / 3.0);
    approx_eq(y[2], 0.0);
    approx_eq(y[3], (2.0 + 8.0) / 2.0);
}

#[test]
fn all_zero_pool_is_negative_for_positive_lod() {
    let phi = PoolingMatrix::from_rows(&[vec![1, 1, 1], vec![1, 0, 0]]).unwrap();
    let x = vec![0.0, 0.0, 0.0];
    for lod in [1e-12, 1.0, 1e6] {
        let (y, ytilde) = measure(&phi, &x, lod).unwrap();
        assert!(y.iter().all(|&yi| yi == 0.0));
        assert!(ytilde.iter().all(|&t| t == 0), "lod={lod}");
    }
}

#[test]
fn mean_exactly_at_lod_is_positive() {
    let phi = PoolingMatrix::from_rows(&[vec![1, 1], vec![1, 0]]).unwrap();
    let x = vec![2.0, 4.0];
    let lod = 3.0;
    let (y, ytilde) = measure(&phi, &x, lod).unwrap();
    approx_eq(y[0], 3.0);
    assert_eq!(ytilde[0], 1);
    approx_eq(y[1], 2.0);
    assert_eq!(ytilde[1], 0);
}

#[test]
fn comp_elimination_drops_samples_in_negative_pools() {
    // Pool 0 negative covers samples 0,1; pool 1 positive covers 2,3.
    let phi = PoolingMatrix::from_rows(&[vec![1, 1, 0, 0], vec![0, 0, 1, 1]]).unwrap();
    let x = vec![0.0, 0.0, 5.0, 0.0];
    let (_y, ytilde) = measure(&phi, &x, 1.0).unwrap();
    assert_eq!(ytilde, vec![0, 1]);
    let n = phi.n();
    let (sk, s2k) = decode(&ytilde, &phi, n).unwrap();
    for j in [0usize, 1] {
        assert!(!sk.contains(&j), "sample {j} in S_hat_k");
        assert!(!s2k.contains(&j), "sample {j} in S_hat_2k");
    }
    assert!(s2k.contains(&2));
}

#[test]
fn dilution_constrained_rows_have_exactly_pr_ones() {
    let mut rng = StdRng::seed_from_u64(7);
    let pr = 5;
    let phi = sample_pooling(
        12,
        20,
        PoolingDesign::DilutionConstrained { pr },
        &mut rng,
    )
    .unwrap();
    assert_eq!(phi.m(), 12);
    assert_eq!(phi.n(), 20);
    for i in 0..phi.m() {
        assert_eq!(phi.row_sum(i), pr, "row {i}");
    }
}

#[test]
fn sample_constrained_columns_have_at_most_pc_ones() {
    let mut rng = StdRng::seed_from_u64(11);
    let pc = 3;
    let phi = sample_pooling(
        15,
        25,
        PoolingDesign::SampleConstrained { pc },
        &mut rng,
    )
    .unwrap();
    for j in 0..phi.n() {
        let ones = phi.col_sum(j);
        assert!(ones <= pc, "column {j} has {ones} ones");
    }
}

#[test]
fn unconstrained_p_one_is_all_ones_p_zero_is_all_zeros() {
    let mut rng = StdRng::seed_from_u64(1);
    let ones = sample_pooling(4, 6, PoolingDesign::Unconstrained { p: 1.0 }, &mut rng)
        .unwrap();
    let zeros = sample_pooling(4, 6, PoolingDesign::Unconstrained { p: 0.0 }, &mut rng)
        .unwrap();
    for i in 0..4 {
        for j in 0..6 {
            assert_eq!(ones.get(i, j), 1);
            assert_eq!(zeros.get(i, j), 0);
        }
    }
}

#[test]
fn perfect_isolation_identity_recovers_single_positive() {
    let n = 8;
    let phi = PoolingMatrix::identity(n);
    let mut x = vec![0.0; n];
    x[3] = 10.0;
    let (_y, ytilde) = measure(&phi, &x, 1e-9).unwrap();
    let (sk, s2k) = decode(&ytilde, &phi, 1).unwrap();
    assert_eq!(sk, vec![3]);
    assert_eq!(s2k, vec![3]);
}

#[test]
fn disjoint_pools_of_size_two_recover_both_positives() {
    // Two disjoint size-2 pools, each with one positive. Companion negatives
    // are cleared by extra negative singleton pools so COMP leaves only the
    // two positives; k=2 recovers both in S_hat_k and S_hat_2k.
    let phi = PoolingMatrix::from_rows(&[
        vec![1, 1, 0, 0],
        vec![0, 0, 1, 1],
        vec![0, 1, 0, 0],
        vec![0, 0, 0, 1],
    ])
    .unwrap();
    let x = vec![10.0, 0.0, 10.0, 0.0];
    let (_y, ytilde) = measure(&phi, &x, 1e-9).unwrap();
    assert_eq!(ytilde[0], 1);
    assert_eq!(ytilde[1], 1);
    assert_eq!(ytilde[2], 0);
    assert_eq!(ytilde[3], 0);
    let (sk, s2k) = decode(&ytilde, &phi, 2).unwrap();
    assert_eq!(sk.len(), 2);
    assert!(sk.contains(&0) && sk.contains(&2));
    assert!(s2k.contains(&0) && s2k.contains(&2));
}

#[test]
fn top_2k_contains_top_k() {
    let phi = PoolingMatrix::from_rows(&[
        vec![1, 1, 1, 0, 0, 0],
        vec![1, 0, 0, 1, 0, 0],
        vec![0, 1, 0, 0, 1, 0],
        vec![0, 0, 0, 0, 0, 1],
    ])
    .unwrap();
    let x = vec![8.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_y, ytilde) = measure(&phi, &x, 1.0).unwrap();
    let (sk, s2k) = decode(&ytilde, &phi, 1).unwrap();
    assert!(s2k.len() >= sk.len());
    for j in &sk {
        assert!(s2k.contains(j), "{j} in S_k but not S_2k");
    }
    let (sk2, s2k2) = decode(&ytilde, &phi, 2).unwrap();
    assert!(s2k2.len() >= sk2.len());
    for j in &sk2 {
        assert!(s2k2.contains(j));
    }
}

#[test]
fn all_negative_covering_phi_empties_estimates() {
    let phi = PoolingMatrix::identity(5);
    let x = vec![0.0; 5];
    let (_y, ytilde) = measure(&phi, &x, 1.0).unwrap();
    assert!(ytilde.iter().all(|&t| t == 0));
    let (sk, s2k) = decode(&ytilde, &phi, 2).unwrap();
    assert!(sk.is_empty());
    assert!(s2k.is_empty());
}

#[test]
fn frequency_ranking_puts_more_positive_pools_first() {
    // Sample 2 appears in two positive pools, sample 0 in one; sample 1 is
    // eliminated by a negative pool. Larger index 2 must rank first.
    let phi = PoolingMatrix::from_rows(&[
        vec![1, 0, 1, 0],
        vec![0, 0, 1, 0],
        vec![0, 1, 0, 0],
    ])
    .unwrap();
    let x = vec![0.0, 0.0, 9.0, 0.0];
    let (_y, ytilde) = measure(&phi, &x, 1.0).unwrap();
    assert_eq!(ytilde, vec![1, 1, 0]);
    let (sk, s2k) = decode(&ytilde, &phi, 1).unwrap();
    assert_eq!(sk[0], 2);
    assert_eq!(s2k[0], 2);
    assert!(!sk.contains(&1));
    assert!(!s2k.contains(&1));
}

#[test]
fn tie_break_prefers_smaller_index() {
    let phi = PoolingMatrix::from_rows(&[vec![1, 1, 0], vec![0, 0, 1]]).unwrap();
    let ytilde = vec![1, 0];
    let (sk, _) = decode(&ytilde, &phi, 2).unwrap();
    // Sample 2 eliminated; 0 and 1 both have B=1; smaller index first.
    assert_eq!(sk, vec![0, 1]);
}
