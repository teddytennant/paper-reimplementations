//! Certificate-governed CRT sparse FFT, Algorithm 1 of arXiv:2609.34086
//! (Multi-Stage Screened Sparse FFT, Global-Label Form).
//!
//! Arithmetic is IEEE-754 `f64` complex, not the paper's exact real-RAM.
//! The occupied equal-magnitude pre-screen uses the nMAD surrogate of
//! Definition VII.1 with an implementer-chosen `tau_cv` (default `1e-8`).
//!
//! DFT convention (Def. III.1 / numpy.fft.fft / rustfft forward):
//! `X[f] = sum_n x[n] exp(-j 2π f n / N)` (unnormalized). The inverse used
//! by the verifier is `x[n] = (1/N) sum_f X[f] exp(+j 2π f n / N)`.

pub mod primes;

use std::collections::BTreeSet;
use std::f64::consts::PI;

pub use num_complex::Complex64;
pub use primes::engineered_primes;

/// Default nMAD acceptance tolerance (Def. VII.1). The paper does not
/// calibrate `τ_cv`; this crate uses `1e-8`.
pub const TAU_CV_DEFAULT: f64 = 1e-8;

/// Root-membership tolerance for `ω^{g0} ≈ w` under f64.
const ROOT_TOL: f64 = 1e-8;

/// Final-verifier residual bound (Def. VI.2), f64 stand-in for exact zero.
const VERIFIER_EPS: f64 = 1e-8;

/// Co-prime-increment shifts `{0, Δ1, Δ1+Δ2}` with `Δ1=31`, `Δ2=32`.
const SHIFTS: [usize; 3] = [0, 31, 63];

/// Result of [`sparse_fft`].
#[derive(Clone, Debug)]
pub struct SparseFftResult {
    /// Length-`N` unnormalized DFT. Off-support entries are zero on the
    /// verified sparse path; otherwise this is the dense DFT of `x`.
    pub spectrum: Vec<Complex64>,
    /// `true` iff the sound final verifier accepted the sparse candidate.
    pub used_sparse: bool,
    /// Global labels accumulated at the point of return (sorted).
    pub candidates: Vec<usize>,
}

/// Multi-stage screened sparse FFT (Algorithm 1).
///
/// `primes` are the stage moduli `p_ℓ`. If `x.len()` is not their product,
/// or the list is not `L ≥ 2` distinct odd primes, the dense DFT of `x` is
/// returned immediately.
pub fn sparse_fft(
    x: &[Complex64],
    k: usize,
    primes: &[usize],
    tau_cv: f64,
) -> SparseFftResult {
    let n = x.len();
    if k < 3 || n == 0 || !primes::is_engineered_length(n, primes) {
        return fallback(x, Vec::new());
    }
    let c = primes::envelope_c(k, primes);

    let mut g_acc: Option<BTreeSet<usize>> = None;
    let mut recovery: Option<(usize, Vec<Complex64>)> = None;

    for &p in primes {
        let d = n / p;
        let mut xp: Vec<Vec<Complex64>> = Vec::with_capacity(3);
        for &sigma in &SHIFTS {
            let mut y = vec![Complex64::new(0.0, 0.0); p];
            for m in 0..p {
                y[m] = x[(sigma + m * d) % n];
            }
            xp.push(dft(&y));
        }
        if recovery.is_none() {
            recovery = Some((p, xp[0].clone()));
        }

        let mut g_l = BTreeSet::new();
        for r in 0..p {
            let z0 = xp[0][r];
            let z31 = xp[1][r];
            let z63 = xp[2][r];
            if !occupied_equal_magnitude_screen(z0, z31, z63, tau_cv) {
                continue;
            }
            if let Some(g) = emit_global_label(z0, z31, z63, r, p, n) {
                g_l.insert(g);
            }
        }

        if g_l.is_empty() || g_l.len() > c.saturating_mul(k) {
            let cand: Vec<usize> = match &g_acc {
                Some(prev) => prev.iter().copied().collect(),
                None => g_l.into_iter().collect(),
            };
            return fallback(x, cand);
        }

        g_acc = Some(match g_acc.take() {
            None => g_l,
            Some(prev) => prev.intersection(&g_l).copied().collect(),
        });
    }

    let g = g_acc.unwrap_or_default();
    let candidates: Vec<usize> = g.iter().copied().collect();
    if g.len() > k {
        return fallback(x, candidates);
    }

    let (p_rec, xp0) = match recovery {
        Some(v) => v,
        None => return fallback(x, candidates),
    };
    let scale = (n as f64) / (p_rec as f64);
    let mut xhat = vec![Complex64::new(0.0, 0.0); n];
    for &freq in &g {
        xhat[freq] = xp0[freq % p_rec].scale(scale);
    }

    if verifier_pass(x, &xhat, &g, k) {
        SparseFftResult {
            spectrum: xhat,
            used_sparse: true,
            candidates,
        }
    } else {
        fallback(x, candidates)
    }
}

fn fallback(x: &[Complex64], candidates: Vec<usize>) -> SparseFftResult {
    SparseFftResult {
        spectrum: dense_fft(x),
        used_sparse: false,
        candidates,
    }
}

/// Unnormalized forward DFT of `x` (numpy / paper Def. III.1).
pub fn dense_fft(x: &[Complex64]) -> Vec<Complex64> {
    dft(x)
}

/// Occupied equal-magnitude pre-screen, nMAD surrogate (Def. V.3(1), VII.1).
///
/// Rejects the bin if any observation is zero (ratio guard) or if
/// `nMAD >= tau_cv`. `median(A) = 0` is also a non-candidate.
pub fn occupied_equal_magnitude_screen(
    z0: Complex64,
    z31: Complex64,
    z63: Complex64,
    tau_cv: f64,
) -> bool {
    let a = [z0.norm(), z31.norm(), z63.norm()];
    if a.iter().any(|&v| v == 0.0) {
        return false;
    }
    match nmad(a) {
        Some(v) => v < tau_cv,
        None => false,
    }
}

/// Normalized MAD of three magnitudes: `MAD(A) / median(A)`.
///
/// `MAD(A) = median({|a - median(A)| : a ∈ A})`. Returns `None` when
/// `median(A) = 0` (all-zero / degenerate occupancy).
pub fn nmad(a: [f64; 3]) -> Option<f64> {
    let med = median3(a);
    if med == 0.0 {
        return None;
    }
    Some(mad3(a) / med)
}

/// Total label-emission rule (Def. V.3 / VI.5).
///
/// `u31 = z31/z0`, `u32 = z63/z31`, `w = u32/u31`,
/// `g0 = round(N arg(w) / 2π) mod N`.
/// Emits `g0` iff `ω^{g0} ≈ w` and `g0 ≡ r (mod p)`. At most one label.
pub fn emit_global_label(
    z0: Complex64,
    z31: Complex64,
    z63: Complex64,
    r: usize,
    p: usize,
    n: usize,
) -> Option<usize> {
    if n == 0 || p == 0 {
        return None;
    }
    if z0.norm() == 0.0 || z31.norm() == 0.0 || z63.norm() == 0.0 {
        return None;
    }
    let u31 = z31 / z0;
    let u32 = z63 / z31;
    let w = u32 / u31;
    let ang = w.arg();
    if !ang.is_finite() {
        return None;
    }
    let g0_raw = (n as f64) * ang / (2.0 * PI);
    let g0 = wrap_mod(g0_raw.round() as i64, n);
    let omega_g = Complex64::from_polar(1.0, 2.0 * PI * (g0 as f64) / (n as f64));
    if (omega_g - w).norm() >= ROOT_TOL {
        return None;
    }
    if g0 % p != r % p {
        return None;
    }
    Some(g0)
}

fn verifier_pass(
    x: &[Complex64],
    xhat: &[Complex64],
    g: &BTreeSet<usize>,
    k: usize,
) -> bool {
    let n = x.len();
    if n == 0 {
        return false;
    }
    let w_len = k.saturating_add(g.len());
    let n_f = n as f64;
    for i in 0..w_len {
        let nn = i % n;
        let mut s = Complex64::new(0.0, 0.0);
        for &f in g {
            let ang = 2.0 * PI * (f as f64) * (nn as f64) / n_f;
            s += xhat[f] * Complex64::from_polar(1.0, ang);
        }
        let xhat_n = s / n_f;
        if (x[nn] - xhat_n).norm() >= VERIFIER_EPS {
            return false;
        }
    }
    true
}

fn median3(mut v: [f64; 3]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

fn mad3(a: [f64; 3]) -> f64 {
    let med = median3(a);
    median3([(a[0] - med).abs(), (a[1] - med).abs(), (a[2] - med).abs()])
}

fn wrap_mod(n: i64, m: usize) -> usize {
    let m = m as i64;
    if m == 0 {
        return 0;
    }
    (((n % m) + m) % m) as usize
}

/// Unnormalized forward DFT. Naive for `n ≤ 48`, mixed-radix Cooley–Tukey
/// otherwise. Matches `numpy.fft.fft`.
fn dft(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    if n <= 48 {
        return dft_naive(x);
    }
    let f = smallest_factor(n);
    if f == n {
        dft_naive(x)
    } else {
        dft_mixed(x, f, n / f)
    }
}

fn dft_naive(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    let n_f = n as f64;
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (f, slot) in out.iter_mut().enumerate() {
        let mut s = Complex64::new(0.0, 0.0);
        for (m, &xm) in x.iter().enumerate() {
            let ang = -2.0 * PI * (f as f64) * (m as f64) / n_f;
            s += xm * Complex64::from_polar(1.0, ang);
        }
        *slot = s;
    }
    out
}

/// Mixed-radix Cooley–Tukey: `N = n1 * n2`, `n = n1 + n1*n2_index`.
fn dft_mixed(x: &[Complex64], n1: usize, n2: usize) -> Vec<Complex64> {
    let n = n1 * n2;
    debug_assert_eq!(x.len(), n);
    let n_f = n as f64;

    // inner[n1_idx][k2] = DFT_{n2} of x[n1_idx + n1 * n2_idx]
    let mut inner = vec![vec![Complex64::new(0.0, 0.0); n2]; n1];
    for n1_idx in 0..n1 {
        let mut col = vec![Complex64::new(0.0, 0.0); n2];
        for n2_idx in 0..n2 {
            col[n2_idx] = x[n1_idx + n1 * n2_idx];
        }
        inner[n1_idx] = dft(&col);
        for (k2, v) in inner[n1_idx].iter_mut().enumerate() {
            let ang = -2.0 * PI * (k2 as f64) * (n1_idx as f64) / n_f;
            *v *= Complex64::from_polar(1.0, ang);
        }
    }

    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k2 in 0..n2 {
        let mut row = vec![Complex64::new(0.0, 0.0); n1];
        for n1_idx in 0..n1 {
            row[n1_idx] = inner[n1_idx][k2];
        }
        let row_hat = dft(&row);
        for (k1, v) in row_hat.into_iter().enumerate() {
            out[k1 * n2 + k2] = v;
        }
    }
    out
}

fn smallest_factor(n: usize) -> usize {
    if n < 2 {
        return n;
    }
    if n % 2 == 0 {
        return 2;
    }
    let mut d: usize = 3;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            return d;
        }
        d += 2;
    }
    n
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn dft_delta_is_ones() {
        let mut x = vec![Complex64::new(0.0, 0.0); 8];
        x[0] = Complex64::new(1.0, 0.0);
        let xhat = dft(&x);
        for v in &xhat {
            assert!((v - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn dft_ones_is_n_at_dc() {
        let x = vec![Complex64::new(1.0, 0.0); 7];
        let xhat = dft(&x);
        assert!((xhat[0] - Complex64::new(7.0, 0.0)).norm() < 1e-12);
        for v in xhat.iter().skip(1) {
            assert!(v.norm() < 1e-12);
        }
    }
}
