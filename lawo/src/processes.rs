//! Synthetic processes from Section 5 and Appendix B.

use rand::Rng;

use crate::LabeledPoint;

/// P(L̃ ≥ r) for L̃ ~ Unif({2,…,ℓ}).
fn tilde_survival(r: usize, ell: usize) -> f64 {
    if r > ell {
        0.0
    } else if r <= 2 {
        1.0
    } else {
        (ell - r + 1) as f64 / (ell - 1) as f64
    }
}

/// Stationary first-block length (footnote 2):
/// `P(L_1 = r) = 2 P(L̃ ≥ r) / (ℓ + 2)` for `r ∈ {1,…,ℓ}`,
/// where `L̃ ~ Unif({2,…,ℓ})`.
pub fn first_block_pmf(r: usize, ell: usize) -> f64 {
    assert!(ell >= 2, "ℓ ≥ 2");
    if r < 1 || r > ell {
        return 0.0;
    }
    2.0 * tilde_survival(r, ell) / (ell + 2) as f64
}

fn sample_first_block_len<R: Rng>(ell: usize, rng: &mut R) -> usize {
    let u: f64 = rng.gen();
    let mut cdf = 0.0;
    for r in 1..=ell {
        cdf += first_block_pmf(r, ell);
        if u <= cdf {
            return r;
        }
    }
    ell
}

/// Repeated-block chain (Section 5.1 / Appendix B).
///
/// Draw a base sequence i.i.d. uniform on `{0,…,k-1}` and repeat the `i`-th
/// symbol `L_i` times, with `L_1` from [`first_block_pmf`] and subsequent
/// `L_i ~ Unif({2,…,ℓ})`, truncating to length `len`.
pub fn repeated_block<R: Rng>(len: usize, k: usize, ell: usize, rng: &mut R) -> Vec<usize> {
    assert!(len >= 1 && k >= 1 && ell >= 2);
    let mut out = Vec::with_capacity(len);
    let mut remaining = sample_first_block_len(ell, rng);
    let mut symbol = rng.gen_range(0..k);
    while out.len() < len {
        if remaining == 0 {
            remaining = rng.gen_range(2..=ell);
            symbol = rng.gen_range(0..k);
        }
        out.push(symbol);
        remaining -= 1;
    }
    out
}

/// Sticky Markov chain of Section 5.1:
/// `P = p 1 π^⊤ + (1 − p) I` with uniform `π` on `{0,…,k-1}`.
///
/// Started from `π` (stationarity). `p` is the resampling probability; the
/// paper's mixing-time parameter satisfies `T_mix ≍ 1/p`.
pub fn sticky_markov<R: Rng>(len: usize, k: usize, p: f64, rng: &mut R) -> Vec<usize> {
    assert!(len >= 1 && k >= 1);
    assert!((0.0..=1.0).contains(&p));
    let mut x = rng.gen_range(0..k);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(x);
        if rng.gen::<f64>() < p {
            x = rng.gen_range(0..k);
        }
    }
    out
}

fn std_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box–Muller
    let u1 = rng.gen::<f64>().max(f64::EPSILON);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn std_normal_vec<R: Rng>(d: usize, rng: &mut R) -> Vec<f64> {
    (0..d).map(|_| std_normal(rng)).collect()
}

fn add_assign(dst: &mut [f64], src: &[f64]) {
    for (a, b) in dst.iter_mut().zip(src.iter()) {
        *a += *b;
    }
}

/// Moving-average process of Section 5.2:
/// `X_i = ∑_{j=0}^q ε_{i+j}` with `ε_t iid N(0, I_d)`.
pub fn moving_average<R: Rng>(len: usize, q: usize, d: usize, rng: &mut R) -> Vec<Vec<f64>> {
    assert!(len >= 1 && d >= 1);
    let n_eps = len + q;
    let eps: Vec<Vec<f64>> = (0..n_eps).map(|_| std_normal_vec(d, rng)).collect();
    let mut xs = Vec::with_capacity(len);
    for i in 0..len {
        let mut x = vec![0.0; d];
        for j in 0..=q {
            add_assign(&mut x, &eps[i + j]);
        }
        xs.push(x);
    }
    xs
}

/// Stationary AR(1) of Section 5.2:
/// `X_i = ϕ X_{i-1} + √(1−ϕ²) ε_i`, `X_0, ε_i iid N(0, I_d)`.
///
/// Returns `(X_1,…,X_len)` (the chain is started at stationarity).
pub fn autoregressive<R: Rng>(len: usize, phi: f64, d: usize, rng: &mut R) -> Vec<Vec<f64>> {
    assert!(len >= 1 && d >= 1);
    assert!(phi.abs() < 1.0);
    let scale = (1.0 - phi * phi).sqrt();
    let mut x = std_normal_vec(d, rng); // X_0
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let eps = std_normal_vec(d, rng);
        x = x
            .iter()
            .zip(eps.iter())
            .map(|(xi, e)| phi * xi + scale * e)
            .collect();
        out.push(x.clone());
    }
    out
}

/// Labels `Y_i = sign(⟨e_1, X_i⟩)` of Section 5.2 (`+1` on `{x_1 ≥ 0}`).
pub fn sign_labels(xs: &[Vec<f64>]) -> Vec<LabeledPoint> {
    xs.iter()
        .map(|x| {
            let y = if x.first().copied().unwrap_or(0.0) >= 0.0 {
                1
            } else {
                -1
            };
            LabeledPoint { x: x.clone(), y }
        })
        .collect()
}
