//! Equation / invariant tests for arXiv:2609.19529.

use lawo::{
    add_constant_unseen, autoregressive, average_skipped, classification_test_error_lawo,
    count_surprise_lawo, deletion_window_1based, first_block_pmf, good_turing, knn_predict,
    leave_a_window_out, leave_one_out, local_count_surprise, local_knn_error, local_nn_tail,
    local_surprise, moving_average, nn_tail_lawo, realized_count_surprise, realized_knn_test_error,
    realized_nn_tail, realized_surprise, repeated_block, sign_labels, singleton_fraction,
    skipped_lawo, sticky_markov, surprise_lawo, LabeledPoint, Window, KNN_K, KT_ALPHA,
    LAPLACE_ALPHA,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
}

/// Eq. (7): `D_i = {k ∈ [i, (i+τ−1) ∧ n]}` (1-based).
#[test]
fn eq7_deletion_window_1based() {
    let n = 10usize;
    let tau = 3usize;
    assert_eq!(deletion_window_1based(1, n, tau), (1, 3));
    assert_eq!(deletion_window_1based(5, n, tau), (5, 7));
    assert_eq!(deletion_window_1based(9, n, tau), (9, 10));
    assert_eq!(deletion_window_1based(10, n, tau), (10, 10));

    // τ = 1 is a singleton {i} for every i (leave-one-out).
    for i in 1..=n {
        assert_eq!(deletion_window_1based(i, n, 1), (i, i));
    }

    // Truncation at the end: (i+τ−1) ∧ n.
    assert_eq!(deletion_window_1based(1, 5, 10), (1, 5));
    assert_eq!(deletion_window_1based(4, 5, 3), (4, 5));
}

/// 0-based [`Window`] matches the 1-based paper formula.
#[test]
fn window_matches_eq7() {
    let n = 7usize;
    for tau in 1..=n {
        for i0 in 0..n {
            let w = Window::deletion(i0, n, tau);
            let (lo1, hi1) = deletion_window_1based(i0 + 1, n, tau);
            assert_eq!(w.lo, lo1 - 1);
            assert_eq!(w.hi, hi1 - 1);
            assert_eq!(w.n_deleted(), hi1 - lo1 + 1);
            assert_eq!(w.n_retained(), n - w.n_deleted());
            // D_i starts at i and runs forward (a window *after* each index,
            // including the index itself).
            assert_eq!(w.lo, i0);
            assert!(w.hi >= w.lo);
            assert!(w.hi < n);
            for j in 0..n {
                let in_d = j + 1 >= lo1 && j + 1 <= hi1;
                assert_eq!(w.deleted(j), in_d);
            }
        }
    }
}

/// τ = 1 reduces to leave-one-out: `D_i = {i}`, `I_i = [n] \ {i}`.
#[test]
fn tau_1_is_leave_one_out() {
    let xs = [3u8, 1, 4, 1, 5, 9, 2];
    let n = xs.len();
    for i in 0..n {
        let w = Window::deletion(i, n, 1);
        assert_eq!(w.lo, i);
        assert_eq!(w.hi, i);
        let retained: Vec<usize> = w.retained_indices().collect();
        let expected: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        assert_eq!(retained, expected);
    }

    fn local(i: usize, xs: &[u8], w: Window) -> f64 {
        // f(X_i; dP_{I_i}) = X_i * (mean of retained points)
        let m = w.n_retained() as f64;
        let mean = w.retained_indices().map(|j| xs[j] as f64).sum::<f64>() / m;
        xs[i] as f64 * mean
    }
    close(leave_a_window_out(&xs, 1, local), leave_one_out(&xs, local));
    close(surprise_lawo(&xs, 1), good_turing(&xs));
}

/// Hand-computed novelty indicators on `X = (A,A,B,C,C)`.
#[test]
fn surprise_hand_computation() {
    // A=0, B=1, C=2
    let xs = [0u8, 0, 1, 2, 2];

    // τ = 1 (LOO / Good–Turing): only B is a singleton.
    // i=0 A: I={A,B,C,C} → 0
    // i=1 A: I={A,B,C,C} → 0
    // i=2 B: I={A,A,C,C} → 1
    // i=3 C: I={A,A,B,C} → 0
    // i=4 C: I={A,A,B,C} → 0
    assert_eq!(local_surprise(&xs, 0, 1), 0.0);
    assert_eq!(local_surprise(&xs, 1, 1), 0.0);
    assert_eq!(local_surprise(&xs, 2, 1), 1.0);
    assert_eq!(local_surprise(&xs, 3, 1), 0.0);
    assert_eq!(local_surprise(&xs, 4, 1), 0.0);
    close(surprise_lawo(&xs, 1), 1.0 / 5.0);
    close(good_turing(&xs), 1.0 / 5.0);
    close(singleton_fraction(&xs), 1.0 / 5.0);

    // τ = 2:
    // i=0 A, D={A,A}, I={B,C,C} → 1
    // i=1 A, D={A,B}, I={A,C,C} → 0
    // i=2 B, D={B,C}, I={A,A,C} → 1
    // i=3 C, D={C,C}, I={A,A,B} → 1
    // i=4 C, D={C},   I={A,A,B,C} → 0
    assert_eq!(local_surprise(&xs, 0, 2), 1.0);
    assert_eq!(local_surprise(&xs, 1, 2), 0.0);
    assert_eq!(local_surprise(&xs, 2, 2), 1.0);
    assert_eq!(local_surprise(&xs, 3, 2), 1.0);
    assert_eq!(local_surprise(&xs, 4, 2), 0.0);
    close(surprise_lawo(&xs, 2), 3.0 / 5.0);
}

/// Count-surprise `I{N_{X_i}(X_{I_i}) ≤ ζ}` (Section 4.2).
#[test]
fn count_surprise_hand_computation() {
    let xs = [0u8, 0, 1, 2, 2];
    // ζ = 0 coincides with surprise.
    close(count_surprise_lawo(&xs, 1, 0), surprise_lawo(&xs, 1));
    close(count_surprise_lawo(&xs, 2, 0), surprise_lawo(&xs, 2));

    // τ = 1, ζ = 1: every local count is 0 or 1, so estimator = 1.
    for i in 0..5 {
        assert_eq!(local_count_surprise(&xs, i, 1, 1), 1.0);
    }
    close(count_surprise_lawo(&xs, 1, 1), 1.0);

    // Realized: next token 1 (B) has count 1 in the train set.
    close(realized_count_surprise(&xs, &1u8, 0), 0.0);
    close(realized_count_surprise(&xs, &1u8, 1), 1.0);
    close(realized_count_surprise(&xs, &7u8, 0), 1.0);
}

/// Nearest-neighbour tail on a 1-d trajectory (Section 4.3).
#[test]
fn nn_tail_hand_computation() {
    let xs = [0.0, 0.1, 5.0, 5.2, 10.0];
    let delta = 0.5;

    // τ = 1: only 10.0 is more than 0.5 from every other point.
    assert_eq!(local_nn_tail(&xs, 0, 1, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 1, 1, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 2, 1, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 3, 1, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 4, 1, delta), 1.0);
    close(nn_tail_lawo(&xs, 1, delta), 0.2);

    // τ = 2:
    // i=0 D={0,1} I={5,5.2,10} min=5 > 0.5 → 1
    // i=1 D={1,2} I={0,5.2,10} min=0.1 ≯ → 0
    // i=2 D={2,3} I={0,0.1,10} min=5 > 0.5 → 1
    // i=3 D={3,4} I={0,0.1,5}  min=0.2 ≯ → 0
    // i=4 D={4}   I={0,0.1,5,5.2} min=4.8 > 0.5 → 1
    assert_eq!(local_nn_tail(&xs, 0, 2, delta), 1.0);
    assert_eq!(local_nn_tail(&xs, 1, 2, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 2, 2, delta), 1.0);
    assert_eq!(local_nn_tail(&xs, 3, 2, delta), 0.0);
    assert_eq!(local_nn_tail(&xs, 4, 2, delta), 1.0);
    close(nn_tail_lawo(&xs, 2, delta), 0.6);

    close(realized_nn_tail(&xs, 10.4, 0.5), 0.0);
    close(realized_nn_tail(&xs, 20.0, 0.5), 1.0);
}

/// 3-NN / 1-NN classification test error (Section 4.4, k of Section 5).
#[test]
fn knn_test_error_hand_computation() {
    let zs = [
        LabeledPoint {
            x: vec![0.0],
            y: -1,
        },
        LabeledPoint { x: vec![1.0], y: 1 },
        LabeledPoint { x: vec![2.0], y: 1 },
        LabeledPoint {
            x: vec![10.0],
            y: -1,
        },
    ];

    // 1-NN, τ = 1:
    // i=0: nearest of {1,2,10} is 1 (+1) vs y=-1 → 1
    // i=1: 0 and 2 are both dist 1; lower index 0 (y=-1) vs y=+1 → 1
    // i=2: nearest 1 (+1) vs +1 → 0
    // i=3: nearest 2 (+1) vs -1 → 1
    assert_eq!(local_knn_error(&zs, 0, 1, 1), 1.0);
    assert_eq!(local_knn_error(&zs, 1, 1, 1), 1.0);
    assert_eq!(local_knn_error(&zs, 2, 1, 1), 0.0);
    assert_eq!(local_knn_error(&zs, 3, 1, 1), 1.0);
    close(classification_test_error_lawo(&zs, 1, 1), 0.75);

    // 3-NN, τ = 1: every local vote disagrees with y.
    for i in 0..4 {
        assert_eq!(local_knn_error(&zs, i, 1, 3), 1.0);
    }
    close(classification_test_error_lawo(&zs, 1, KNN_K), 1.0);

    // 3-NN, τ = 2:
    // i=0 I={2,3} labels +1,-1; tie → nearest to 0 is 2 (+1) vs -1 → 1
    // i=1 I={0,3} labels -1,-1 → -1 vs +1 → 1
    // i=2 I={0,1} labels -1,+1; tie → nearest to 2 is 1 (+1) vs +1 → 0
    // i=3 I={0,1,2} labels -1,+1,+1 → +1 vs -1 → 1
    assert_eq!(local_knn_error(&zs, 0, 2, 3), 1.0);
    assert_eq!(local_knn_error(&zs, 1, 2, 3), 1.0);
    assert_eq!(local_knn_error(&zs, 2, 2, 3), 0.0);
    assert_eq!(local_knn_error(&zs, 3, 2, 3), 1.0);
    close(classification_test_error_lawo(&zs, 2, 3), 0.75);

    let train = zs[..3].to_vec();
    let test = zs[3].clone();
    close(realized_knn_test_error(&train, &test, 1), 1.0);
}

/// Empty retained set: novelty / infinite NN distance / empty-train k-NN.
#[test]
fn empty_retained_conventions() {
    let xs = [7u8];
    close(surprise_lawo(&xs, 1), 1.0);
    close(count_surprise_lawo(&xs, 1, 0), 1.0);
    close(nn_tail_lawo(&[1.0], 1, 0.0), 1.0);
    close(realized_surprise::<u8>(&[], &0), 1.0);

    let z = [LabeledPoint {
        x: vec![0.0],
        y: -1,
    }];
    // Empty train → predict +1 ≠ −1.
    assert_eq!(knn_predict(&[], &z[0].x, 3), 1);
    close(classification_test_error_lawo(&z, 1, 3), 1.0);
}

/// Eq. (16)–(17): averaging skipped estimators recovers Eq. (8).
#[test]
fn skipped_average_recovers_lawo() {
    let xs = [0u8, 1, 0, 1, 2, 0];
    let tau = 2usize;
    assert_eq!(xs.len() % tau, 0);

    let avg = average_skipped(&xs, tau, |i, xs, _w| local_surprise(xs, i, tau));
    close(avg, surprise_lawo(&xs, tau));

    // Direct check of the 1-based index set τi − u.
    // n=6, τ=2, n0=3.
    // u=0: indices 2,4,6 (1-based) → 1,3,5
    // u=1: indices 1,3,5 → 0,2,4
    let s0 = skipped_lawo(&xs, tau, 0, |i, xs, _w| local_surprise(xs, i, tau));
    let s1 = skipped_lawo(&xs, tau, 1, |i, xs, _w| local_surprise(xs, i, tau));
    let manual0 =
        (local_surprise(&xs, 1, tau) + local_surprise(&xs, 3, tau) + local_surprise(&xs, 5, tau))
            / 3.0;
    let manual1 =
        (local_surprise(&xs, 0, tau) + local_surprise(&xs, 2, tau) + local_surprise(&xs, 4, tau))
            / 3.0;
    close(s0, manual0);
    close(s1, manual1);
    close((s0 + s1) / 2.0, surprise_lawo(&xs, tau));
}

/// Footnote 2: first-block length is a PMF and `E[L_1] = (ℓ+2)/2` wait —
/// actually `∑_r P(L̃ ≥ r) = E[L̃] = (ℓ+2)/2`, so the PMF sums to 1.
#[test]
fn first_block_pmf_is_probability() {
    for ell in 2..=16 {
        let mut s = 0.0;
        for r in 1..=ell {
            s += first_block_pmf(r, ell);
        }
        close(s, 1.0);
        // P(L_1 = 1) = 2/(ℓ+2) (Appendix B).
        close(first_block_pmf(1, ell), 2.0 / (ell + 2) as f64);
    }
}

/// Appendix B: on a repeated-block chain with large alphabet, Good–Turing
/// (LOO, τ=1) is biased low of order 1/ℓ, while LAWO with `τ = ℓ` tracks the
/// true next-token surprise (order `2/(ℓ+2)`).
#[test]
fn repeated_block_loo_biased_lawo_tracks_surprise() {
    let n = 36usize;
    let ell = 3usize;
    let k = 2000usize;
    let trials = 80usize;
    let mut rng = StdRng::seed_from_u64(0);

    let mut sum_gt = 0.0;
    let mut sum_lawo = 0.0;
    let mut sum_true = 0.0;
    for _ in 0..trials {
        let seq = repeated_block(n + 1, k, ell, &mut rng);
        let train = &seq[..n];
        let next = seq[n];
        sum_gt += good_turing(train);
        sum_lawo += surprise_lawo(train, ell);
        sum_true += realized_surprise(train, &next);
    }
    let mean_gt = sum_gt / trials as f64;
    let mean_lawo = sum_lawo / trials as f64;
    let mean_true = sum_true / trials as f64;

    // GT ≤ 2/n in the Appendix B argument; with tiny n we only need a
    // clear gap. True S is about 2/(ℓ+2) = 0.4.
    assert!(
        mean_gt < 0.15,
        "Good–Turing / LOO should be biased low, got {mean_gt}"
    );
    assert!(
        mean_true > 0.25,
        "true surprise should stay order 1/ℓ, got {mean_true}"
    );
    assert!(
        mean_lawo > mean_gt + 0.15,
        "LAWO(τ=ℓ) should exceed biased LOO: LAWO={mean_lawo} GT={mean_gt}"
    );
    assert!(
        (mean_lawo - mean_true).abs() < (mean_gt - mean_true).abs(),
        "LAWO closer to truth than LOO: LAWO={mean_lawo} GT={mean_gt} S={mean_true}"
    );
    assert!(mean_lawo.is_finite() && mean_gt.is_finite());
}

/// Sticky Markov chain (Section 5.1): estimator is in [0,1] and finite;
/// high stickiness makes LOO smaller than a larger window.
#[test]
fn sticky_markov_lawo_finite_and_loo_underestimates() {
    let n = 40usize;
    let k = 500usize;
    let p = 0.15; // T_mix ≍ 1/p
    let mut rng = StdRng::seed_from_u64(1);
    let mut sum_loo = 0.0;
    let mut sum_lawo = 0.0;
    let trials = 40usize;
    for _ in 0..trials {
        let seq = sticky_markov(n, k, p, &mut rng);
        assert_eq!(seq.len(), n);
        assert!(seq.iter().all(|&x| x < k));
        let loo = surprise_lawo(&seq, 1);
        let lawo = surprise_lawo(&seq, 8);
        assert!((0.0..=1.0).contains(&loo));
        assert!((0.0..=1.0).contains(&lawo));
        assert!(loo.is_finite() && lawo.is_finite());
        sum_loo += loo;
        sum_lawo += lawo;
    }
    let mean_loo = sum_loo / trials as f64;
    let mean_lawo = sum_lawo / trials as f64;
    assert!(
        mean_lawo >= mean_loo,
        "larger τ should not be more optimistic about novelty under stickiness: LOO={mean_loo} LAWO={mean_lawo}"
    );
}

/// MA / AR sequences of Section 5.2, tiny n, 3-NN test error in [0,1].
#[test]
fn ma_ar_knn_lawo_finite() {
    let mut rng = StdRng::seed_from_u64(2);
    let n = 24usize;
    let d = 3usize;

    let ma = moving_average(n + 1, 2, d, &mut rng);
    assert_eq!(ma.len(), n + 1);
    assert!(ma.iter().all(|x| x.len() == d));
    let zs = sign_labels(&ma[..n]);
    let te1 = classification_test_error_lawo(&zs, 1, KNN_K);
    let te_tau = classification_test_error_lawo(&zs, 3, KNN_K);
    assert!((0.0..=1.0).contains(&te1) && te1.is_finite());
    assert!((0.0..=1.0).contains(&te_tau) && te_tau.is_finite());
    close(te1, classification_test_error_lawo(&zs, 1, KNN_K));
    let realized = realized_knn_test_error(&zs, &sign_labels(&ma[n..n + 1])[0], KNN_K);
    assert!(realized == 0.0 || realized == 1.0);

    let ar = autoregressive(n + 1, 0.5, d, &mut rng);
    assert_eq!(ar.len(), n + 1);
    let zs = sign_labels(&ar[..n]);
    let te = classification_test_error_lawo(&zs, 4, KNN_K);
    assert!((0.0..=1.0).contains(&te) && te.is_finite());

    // 1-d NN tail on the first coordinate of an MA path.
    let xs: Vec<f64> = ma[..n].iter().map(|x| x[0]).collect();
    let ns = nn_tail_lawo(&xs, 2, 0.25);
    assert!((0.0..=1.0).contains(&ns) && ns.is_finite());
}

/// Add-constant baselines (Section 5.1) are in (0,1) on a large alphabet.
#[test]
fn add_constant_baselines() {
    let xs = [0usize, 0, 1, 2, 2];
    let k = 20usize;
    let lap = add_constant_unseen(&xs, k, LAPLACE_ALPHA);
    let kt = add_constant_unseen(&xs, k, KT_ALPHA);
    assert!(lap > 0.0 && lap < 1.0);
    assert!(kt > 0.0 && kt < 1.0);
    // Three observed types, n=5: Laplace = 1*(20-3)/(5+20) = 17/25.
    close(lap, 17.0 / 25.0);
    close(kt, 0.5 * 17.0 / (5.0 + 0.5 * 20.0));
}

/// Eq. (8) averages n local values; empty input is 0.
#[test]
fn averaging_and_empty() {
    close(surprise_lawo::<u8>(&[], 1), 0.0);
    close(leave_a_window_out::<u8, _>(&[], 3, |_, _, _| 1.0), 0.0);

    let xs = [1u8, 2, 3, 4];
    let ones = leave_a_window_out(&xs, 2, |_, _, _| 1.0);
    close(ones, 1.0);
    let idx_mean = leave_a_window_out(&xs, 2, |i, _, _| i as f64);
    close(idx_mean, (0 + 1 + 2 + 3) as f64 / 4.0);
}

/// Realized surprise on i.i.d. tokens with known alphabet.
#[test]
fn realized_novelty_indicator() {
    let train = [1u8, 1, 2];
    close(realized_surprise(&train, &1), 0.0);
    close(realized_surprise(&train, &2), 0.0);
    close(realized_surprise(&train, &3), 1.0);
}

/// τ=1 surprise equals the singleton fraction for every sequence.
#[test]
fn good_turing_is_singleton_fraction() {
    for seq in [
        vec![0u8],
        vec![0, 0, 0],
        vec![0, 1, 2, 3],
        vec![5, 5, 6, 6, 7],
        vec![1, 2, 1, 2, 1, 3],
    ] {
        close(good_turing(&seq), singleton_fraction(&seq));
        close(surprise_lawo(&seq, 1), singleton_fraction(&seq));
    }
}
