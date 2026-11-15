//! CPU tests of Algorithm 1 on noiseless on-grid k-sparse spectra.

use std::f64::consts::PI;

use crt_sparse_fft::{
    dense_fft, emit_global_label, engineered_primes, nmad,
    occupied_equal_magnitude_screen, sparse_fft, Complex64, TAU_CV_DEFAULT,
};

fn time_signal(n: usize, tones: &[(usize, Complex64)]) -> Vec<Complex64> {
    let n_f = n as f64;
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for (i, slot) in x.iter_mut().enumerate() {
        let mut s = Complex64::new(0.0, 0.0);
        for &(f, c) in tones {
            let ang = 2.0 * PI * (f as f64) * (i as f64) / n_f;
            s += c * Complex64::from_polar(1.0, ang);
        }
        *slot = s / n_f;
    }
    x
}

fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(u, v)| (u - v).norm())
        .fold(0.0, f64::max)
}

fn support_of(spec: &[Complex64], tol: f64) -> Vec<usize> {
    spec.iter()
        .enumerate()
        .filter(|(_, v)| v.norm() > tol)
        .map(|(i, _)| i)
        .collect()
}

fn singleton_obs(
    n: usize,
    p: usize,
    g: usize,
    coeff: Complex64,
) -> (Complex64, Complex64, Complex64) {
    let scale = (p as f64) / (n as f64);
    let z = |sigma: usize| {
        scale
            * coeff
            * Complex64::from_polar(
                1.0,
                2.0 * PI * (sigma as f64) * (g as f64) / (n as f64),
            )
    };
    (z(0), z(31), z(63))
}

#[test]
fn engineered_primes_paper_example() {
    let p = engineered_primes(8, 4).expect("four primes in a widened [8, c*8]");
    assert_eq!(p, vec![11, 13, 17, 19]);
}

#[test]
fn three_singleton_tones_n143() {
    let n = 143;
    let k = 3;
    let primes = [11usize, 13];
    // Distinct residues mod 11 and mod 13.
    let tones = [
        (1usize, Complex64::new(1.2, -0.4)),
        (2, Complex64::new(-0.7, 0.9)),
        (3, Complex64::new(2.0, 0.0)),
    ];
    let x = time_signal(n, &tones);
    let dense = dense_fft(&x);
    let out = sparse_fft(&x, k, &primes, TAU_CV_DEFAULT);
    assert!(out.used_sparse, "all-stage singletons should take the sparse path");
    assert!(
        max_abs_diff(&out.spectrum, &dense) < 1e-6,
        "max abs err {}",
        max_abs_diff(&out.spectrum, &dense)
    );
    let got = support_of(&out.spectrum, 1e-6);
    let mut want: Vec<usize> = tones.iter().map(|t| t.0).collect();
    want.sort_unstable();
    assert_eq!(got, want);
    for &(f, c) in &tones {
        assert!(
            (out.spectrum[f] - c).norm() < 1e-6,
            "coeff at {f}: got {} want {}",
            out.spectrum[f],
            c
        );
    }
}

#[test]
fn paper_example_n46189_tone_67() {
    let n = 46_189;
    let k = 8;
    let primes = [11usize, 13, 17, 19];
    assert_eq!(primes.iter().product::<usize>(), n);
    // f=67 plus singleton-friendly tones (distinct residues at every p_ℓ;
    // 10 collides with 67 mod 19, so it is not used).
    let tones = [
        (67usize, Complex64::new(1.0, 0.0)),
        (4, Complex64::new(2.0, -0.5)),
        (5, Complex64::new(0.5, 0.25)),
        (8, Complex64::new(-1.1, 0.3)),
    ];
    let x = time_signal(n, &tones);
    let dense = dense_fft(&x);
    let out = sparse_fft(&x, k, &primes, TAU_CV_DEFAULT);
    assert!(out.used_sparse, "paper walkthrough support should verify");
    assert!(out.candidates.contains(&67));
    assert!((out.spectrum[67] - Complex64::new(1.0, 0.0)).norm() < 1e-6);
    assert!(max_abs_diff(&out.spectrum, &dense) < 1e-6);
    for &(f, c) in &tones {
        assert!(
            (out.spectrum[f] - c).norm() < 1e-6,
            "coeff at {f}: got {} want {}",
            out.spectrum[f],
            c
        );
    }
}

#[test]
fn collision_falls_back_to_dense() {
    let n = 143;
    let k = 3;
    let primes = [11usize, 13];
    // 1 and 12 share class 1 mod 11; 3 is a singleton at both stages.
    let tones = [
        (1usize, Complex64::new(1.0, 0.0)),
        (12, Complex64::new(1.0, 0.0)),
        (3, Complex64::new(1.0, 0.0)),
    ];
    let x = time_signal(n, &tones);
    let dense = dense_fft(&x);
    let out = sparse_fft(&x, k, &primes, TAU_CV_DEFAULT);
    assert!(
        !out.used_sparse,
        "a true-tone collision must FAIL the verifier or hit a fallback"
    );
    assert!(max_abs_diff(&out.spectrum, &dense) < 1e-6);
}

#[test]
fn zero_signal_is_all_zeros() {
    let n = 143;
    let k = 3;
    let primes = [11usize, 13];
    let x = vec![Complex64::new(0.0, 0.0); n];
    let out = sparse_fft(&x, k, &primes, TAU_CV_DEFAULT);
    assert!(!out.used_sparse, "zero spectrum has no occupied bins");
    assert!(out.spectrum.iter().all(|v| v.norm() < 1e-12));
}

#[test]
fn non_product_length_dense_fallback() {
    let n = 64;
    let k = 3;
    let primes = [11usize, 13];
    let tones = [
        (3usize, Complex64::new(1.0, 0.0)),
        (7, Complex64::new(0.5, -0.2)),
        (11, Complex64::new(-0.3, 0.8)),
    ];
    let x = time_signal(n, &tones);
    let dense = dense_fft(&x);
    let out = sparse_fft(&x, k, &primes, TAU_CV_DEFAULT);
    assert!(!out.used_sparse);
    assert!(max_abs_diff(&out.spectrum, &dense) < 1e-8);
    for &(f, c) in &tones {
        assert!((out.spectrum[f] - c).norm() < 1e-8);
    }
}

#[test]
fn nmad_rejects_unequal_magnitudes() {
    let equal = nmad([1.0, 1.0, 1.0]).unwrap();
    assert!(equal < 1e-15);

    let unequal = nmad([1.0, 2.0, 3.0]).unwrap();
    assert!(
        (unequal - 0.5).abs() < 1e-12,
        "median=2, MAD=1, nMAD=0.5, got {unequal}"
    );
    assert!(unequal >= TAU_CV_DEFAULT);

    let z_eq = Complex64::new(1.0, 0.0);
    assert!(occupied_equal_magnitude_screen(
        z_eq,
        z_eq,
        z_eq,
        TAU_CV_DEFAULT
    ));

    let z1 = Complex64::new(1.0, 0.0);
    let z2 = Complex64::new(2.0, 0.0);
    let z3 = Complex64::new(3.0, 0.0);
    assert!(!occupied_equal_magnitude_screen(
        z1,
        z2,
        z3,
        TAU_CV_DEFAULT
    ));

    let z0 = Complex64::new(0.0, 0.0);
    assert!(!occupied_equal_magnitude_screen(
        z1,
        z0,
        z1,
        TAU_CV_DEFAULT
    ));
}

#[test]
fn label_emission_true_g_and_class_reject() {
    let n = 46_189;
    let p = 11usize;
    let g = 67usize;
    let (z0, z31, z63) = singleton_obs(n, p, g, Complex64::new(1.0, 0.0));
    let r = g % p;
    assert_eq!(r, 1);
    assert_eq!(emit_global_label(z0, z31, z63, r, p, n), Some(67));
    assert_eq!(
        emit_global_label(z0, z31, z63, 0, p, n),
        None,
        "class-inconsistent residue must be rejected"
    );
}
