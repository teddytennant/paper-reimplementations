//! Leave-a-window-out next-token functional estimation
//! (Nakul, Muthukumar, Pananjady, [arXiv:2609.19529](https://arxiv.org/abs/2609.19529)).
//!
//! Observe `X^n = (X_1,…,X_n)` and estimate `θ_f = E[f(X_{n+1}; dP_n)]` (Eq. 3).
//! Classical leave-one-out (Eq. 2) is the special case `τ = 1` of the
//! leave-a-window-out estimator (Eq. 8): for each index `i`, delete the
//! window `D_i` of Eq. (7) and evaluate `f` at `X_i` against the empirical
//! measure of the retained coordinates `I_i = [n] \ D_i`.

use std::collections::HashSet;
use std::hash::Hash;

pub mod processes;

pub use processes::{
    autoregressive, first_block_pmf, moving_average, repeated_block, sign_labels, sticky_markov,
};

/// k-nearest-neighbour rule used in Section 5 (`k = 3`).
pub const KNN_K: usize = 3;

/// Laplace add-constant `α = 1`.
pub const LAPLACE_ALPHA: f64 = 1.0;
/// Krichevsky–Trofimov add-constant `α = 1/2`.
pub const KT_ALPHA: f64 = 0.5;

/// 0-based deletion window `D_i` of Eq. (7).
///
/// Paper (1-based): `D_i = {k ∈ [i, (i+τ-1) ∧ n]}`.
/// Here indices run over `0..n`, so `D_i = {i, …, min(i+τ-1, n-1)}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Proxy test index (0-based).
    pub i: usize,
    /// Inclusive start of `D_i`.
    pub lo: usize,
    /// Inclusive end of `D_i`.
    pub hi: usize,
    /// Trajectory length `n`.
    pub n: usize,
}

impl Window {
    /// Eq. (7), 0-based.
    pub fn deletion(i: usize, n: usize, tau: usize) -> Self {
        assert!(n >= 1, "n ≥ 1");
        assert!(i < n, "i < n");
        assert!(tau >= 1, "τ ≥ 1");
        let lo = i;
        let hi = i.saturating_add(tau - 1).min(n - 1);
        Self { i, lo, hi, n }
    }

    /// Whether `j` belongs to `D_i`.
    pub fn deleted(self, j: usize) -> bool {
        j >= self.lo && j <= self.hi
    }

    /// `I_i = [n] \ D_i`, 0-based.
    pub fn retained_indices(self) -> impl Iterator<Item = usize> {
        (0..self.n).filter(move |&j| !self.deleted(j))
    }

    /// `|D_i|`.
    pub fn n_deleted(self) -> usize {
        self.hi - self.lo + 1
    }

    /// `|I_i|`.
    pub fn n_retained(self) -> usize {
        self.n - self.n_deleted()
    }
}

/// 1-based deletion window of Eq. (7): `(lo, hi)` inclusive, both in `{1,…,n}`.
pub fn deletion_window_1based(i: usize, n: usize, tau: usize) -> (usize, usize) {
    assert!(i >= 1 && i <= n, "i ∈ [n]");
    assert!(tau >= 1, "τ ≥ 1");
    let hi = (i + tau - 1).min(n);
    (i, hi)
}

/// Leave-a-window-out estimator (Eq. 8):
///
/// `θ̂(X^n, τ) = (1/n) ∑_{i=1}^n f(X_i; dP_{I_i})`.
///
/// The closure receives the 0-based index `i`, the full trajectory, and the
/// deletion window `D_i`. An empty trajectory yields `0`.
pub fn leave_a_window_out<T, F>(xs: &[T], tau: usize, mut local: F) -> f64
where
    F: FnMut(usize, &[T], Window) -> f64,
{
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    assert!(tau >= 1, "τ ≥ 1");
    let mut acc = 0.0;
    for i in 0..n {
        let w = Window::deletion(i, n, tau);
        acc += local(i, xs, w);
    }
    acc / n as f64
}

/// Classical leave-one-out (Eq. 2): Eq. (8) at `τ = 1`.
pub fn leave_one_out<T, F>(xs: &[T], local: F) -> f64
where
    F: FnMut(usize, &[T], Window) -> f64,
{
    leave_a_window_out(xs, 1, local)
}

/// Skipped estimator of Eq. (16).
///
/// Requires `n` divisible by `τ`. For offset `u ∈ {0,…,τ-1}` averages the
/// local estimators at 1-based indices `τ i − u` for `i = 1,…,n/τ`.
pub fn skipped_lawo<T, F>(xs: &[T], tau: usize, u: usize, mut local: F) -> f64
where
    F: FnMut(usize, &[T], Window) -> f64,
{
    let n = xs.len();
    assert!(tau >= 1, "τ ≥ 1");
    assert!(n % tau == 0, "n must be divisible by τ");
    assert!(u < tau, "u in 0..tau");
    let n0 = n / tau;
    if n0 == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 1..=n0 {
        let idx = tau * i - u - 1; // 0-based
        let w = Window::deletion(idx, n, tau);
        acc += local(idx, xs, w);
    }
    acc / n0 as f64
}

/// Average of skipped estimators (Eq. 17): `θ̂(τ) = (1/τ) ∑_u θ̂(τ; u)`.
pub fn average_skipped<T, F>(xs: &[T], tau: usize, mut local: F) -> f64
where
    F: FnMut(usize, &[T], Window) -> f64,
{
    let n = xs.len();
    assert!(tau >= 1 && n % tau == 0);
    let mut acc = 0.0;
    for u in 0..tau {
        acc += skipped_lawo(xs, tau, u, &mut local);
    }
    acc / tau as f64
}

// ---------------------------------------------------------------------------
// Functionals (Section 4)
// ---------------------------------------------------------------------------

/// Local novelty indicator `I{X_i ∉ X_{I_i}}` (Eq. 9 / Section 4.1).
pub fn local_surprise<T: Eq>(xs: &[T], i: usize, tau: usize) -> f64 {
    let n = xs.len();
    let w = Window::deletion(i, n, tau);
    let xi = &xs[i];
    if w.retained_indices().any(|j| xs[j] == *xi) {
        0.0
    } else {
        1.0
    }
}

/// Leave-a-window-out surprise-probability estimator `Ŝ(τ)` (Eq. 9).
pub fn surprise_lawo<T: Eq>(xs: &[T], tau: usize) -> f64 {
    leave_a_window_out(xs, tau, |i, xs, _w| local_surprise(xs, i, tau))
}

/// Good–Turing / leave-one-out surprise: `Ŝ(1)`.
pub fn good_turing<T: Eq>(xs: &[T]) -> f64 {
    surprise_lawo(xs, 1)
}

/// Realized surprise `I{X_{n+1} ∉ {X_1,…,X_n}}`.
pub fn realized_surprise<T: Eq>(train: &[T], next: &T) -> f64 {
    if train.iter().any(|x| x == next) {
        0.0
    } else {
        1.0
    }
}

/// Occurrence count `N_x(X_I)`.
pub fn occurrence<T: Eq>(xs: &[T], x: &T, indices: impl Iterator<Item = usize>) -> usize {
    indices.filter(|&j| xs[j] == *x).count()
}

/// Local count-surprise `I{N_{X_i}(X_{I_i}) ≤ ζ}` (Section 4.2).
pub fn local_count_surprise<T: Eq>(xs: &[T], i: usize, tau: usize, zeta: usize) -> f64 {
    let n = xs.len();
    let w = Window::deletion(i, n, tau);
    let c = occurrence(xs, &xs[i], w.retained_indices());
    if c <= zeta {
        1.0
    } else {
        0.0
    }
}

/// Leave-a-window-out count-surprise estimator `Ŝ_{≤ζ}(τ)`.
pub fn count_surprise_lawo<T: Eq>(xs: &[T], tau: usize, zeta: usize) -> f64 {
    leave_a_window_out(xs, tau, |i, xs, _w| local_count_surprise(xs, i, tau, zeta))
}

/// Realized count-surprise `I{N_{X_{n+1}}(X^n) ≤ ζ}`.
pub fn realized_count_surprise<T: Eq>(train: &[T], next: &T, zeta: usize) -> f64 {
    let c = train.iter().filter(|x| *x == next).count();
    if c <= zeta {
        1.0
    } else {
        0.0
    }
}

/// Local nearest-neighbour tail `I{min_{j∈I_i} |X_i − X_j| > δ}` (Section 4.3).
///
/// An empty retained set is treated as min-distance `+∞` (indicator 1).
pub fn local_nn_tail(xs: &[f64], i: usize, tau: usize, delta: f64) -> f64 {
    let n = xs.len();
    let w = Window::deletion(i, n, tau);
    let xi = xs[i];
    let mut any = false;
    let mut min_d = f64::INFINITY;
    for j in w.retained_indices() {
        any = true;
        min_d = min_d.min((xi - xs[j]).abs());
    }
    if !any || min_d > delta {
        1.0
    } else {
        0.0
    }
}

/// Leave-a-window-out nearest-neighbour tail estimator `NŜ(τ)`.
pub fn nn_tail_lawo(xs: &[f64], tau: usize, delta: f64) -> f64 {
    leave_a_window_out(xs, tau, |i, xs, _w| local_nn_tail(xs, i, tau, delta))
}

/// Realized NN tail `I{min_{j∈[n]} |X_{n+1} − X_j| > δ}`.
pub fn realized_nn_tail(train: &[f64], next: f64, delta: f64) -> f64 {
    if train.is_empty() {
        return 1.0;
    }
    let min_d = train
        .iter()
        .map(|x| (next - x).abs())
        .fold(f64::INFINITY, f64::min);
    if min_d > delta {
        1.0
    } else {
        0.0
    }
}

/// Covariate / label pair `Z_i = (X_i, Y_i)` with `Y_i ∈ {±1}` (Section 4.4).
#[derive(Clone, Debug, PartialEq)]
pub struct LabeledPoint {
    pub x: Vec<f64>,
    pub y: i8,
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| {
            let d = u - v;
            d * d
        })
        .sum()
}

/// k-NN majority vote on `train` at `query`.
///
/// Neighbours are ordered by `(squared Euclidean distance, index)`. If the
/// vote sum is 0, the nearest neighbour's label is returned. An empty
/// training set returns `+1`.
pub fn knn_predict(train: &[(usize, &LabeledPoint)], query: &[f64], k: usize) -> i8 {
    if train.is_empty() {
        return 1;
    }
    let mut dists: Vec<(f64, usize, i8)> = train
        .iter()
        .map(|&(idx, z)| (sq_dist(&z.x, query), idx, z.y))
        .collect();
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let take = k.min(dists.len()).max(1);
    let sum: i32 = dists[..take].iter().map(|t| t.2 as i32).sum();
    if sum > 0 {
        1
    } else if sum < 0 {
        -1
    } else {
        dists[0].2
    }
}

/// 0-1 loss of a k-NN classifier trained on `I_i` and evaluated at `Z_i`.
pub fn local_knn_error(zs: &[LabeledPoint], i: usize, tau: usize, k: usize) -> f64 {
    let n = zs.len();
    let w = Window::deletion(i, n, tau);
    let train: Vec<(usize, &LabeledPoint)> = w.retained_indices().map(|j| (j, &zs[j])).collect();
    let pred = knn_predict(&train, &zs[i].x, k);
    if pred == zs[i].y {
        0.0
    } else {
        1.0
    }
}

/// Leave-a-window-out classification test-error estimator `TÊ(τ)` (Section 4.4)
/// with the k-NN rule of Section 5.
pub fn classification_test_error_lawo(zs: &[LabeledPoint], tau: usize, k: usize) -> f64 {
    leave_a_window_out(zs, tau, |i, zs, _w| local_knn_error(zs, i, tau, k))
}

/// Realized 0-1 test error of k-NN trained on `train` and evaluated at `test`.
pub fn realized_knn_test_error(train: &[LabeledPoint], test: &LabeledPoint, k: usize) -> f64 {
    let pairs: Vec<(usize, &LabeledPoint)> =
        train.iter().enumerate().map(|(j, z)| (j, z)).collect();
    let pred = knn_predict(&pairs, &test.x, k);
    if pred == test.y {
        0.0
    } else {
        1.0
    }
}

/// Add-constant predictive mass on unseen symbols, known alphabet size `k`:
/// `α (k − #observed types) / (n + α k)`.
///
/// Laplace uses [`LAPLACE_ALPHA`]; Krichevsky–Trofimov uses [`KT_ALPHA`].
pub fn add_constant_unseen<T: Eq + Hash>(xs: &[T], alphabet: usize, alpha: f64) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let seen = xs.iter().collect::<HashSet<_>>().len();
    let unseen = alphabet.saturating_sub(seen) as f64;
    alpha * unseen / (n + alpha * alphabet as f64)
}

/// Fraction of symbols that appear exactly once (equals Good–Turing / `Ŝ(1)`).
pub fn singleton_fraction<T: Eq>(xs: &[T]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    for i in 0..n {
        let c = xs.iter().filter(|x| **x == xs[i]).count();
        if c == 1 {
            hits += 1;
        }
    }
    hits as f64 / n as f64
}
