use crate::matroid::IndependenceOracle;

/// Rejected caller or internal epsilon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuctionError {
    EpsilonOutOfRange,
}

impl std::fmt::Display for AuctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuctionError::EpsilonOutOfRange => write!(f, "epsilon out of range"),
        }
    }
}

impl std::error::Error for AuctionError {}

/// Recorded common independent set of maximum weight, with the prices and
/// utilities from the phase that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    pub set: Vec<usize>,
    pub weight: f64,
    pub pi: Vec<f64>,
    pub kappa: Vec<f64>,
    /// Phase index of the recorded set. 0 is the initial empty set.
    pub phase: usize,
}

/// Number of phases: ceil(10 * eps^{-2} * ln(2n)).
pub fn num_phases(n: usize, eps: f64) -> usize {
    if n == 0 {
        return 0;
    }
    let t = 10.0 * eps.powi(-2) * (2.0 * (n as f64)).ln();
    t.ceil() as usize
}

fn lambda_at(big_w: f64, eps: f64, k: i64) -> f64 {
    big_w * (1.0 + eps).powf(-(k as f64))
}

/// Smallest element of Lambda = { W * (1+eps)^{-k} | k = 0, 1, 2, ... }
/// that is at least `u`. Returns 0 when `u <= 0`.
pub fn ceil_lambda(u: f64, eps: f64, big_w: f64) -> f64 {
    if u <= 0.0 {
        return 0.0;
    }
    assert!(eps > 0.0 && big_w > 0.0, "eps and W must be positive");
    if u > big_w {
        // Lambda is bounded by W. The auction never asks for a ceiling above W.
        return lambda_at(big_w, eps, 0);
    }
    let k_real = (big_w / u).ln() / (1.0 + eps).ln();
    let mut k = k_real.floor() as i64;
    if k < 0 {
        k = 0;
    }
    // Largest k with lambda(k) >= u, repaired for rounding at bucket edges.
    let mut guard = 0;
    while k > 0 && lambda_at(big_w, eps, k) < u && guard < 16 {
        k -= 1;
        guard += 1;
    }
    guard = 0;
    while k < 1_000_000 && guard < 16 {
        let prev = lambda_at(big_w, eps, k + 1);
        if prev < u {
            break;
        }
        k += 1;
        guard += 1;
    }
    let mut lambda = lambda_at(big_w, eps, k);
    if lambda < u && k > 0 {
        k -= 1;
        lambda = lambda_at(big_w, eps, k);
    }
    lambda
}

fn drop_prefix_independent<O: IndependenceOracle>(
    oracle: &O,
    sorted: &[usize],
    prefix_len: usize,
    e: usize,
) -> bool {
    let mut set = Vec::with_capacity(sorted.len() - prefix_len + 1);
    set.extend_from_slice(&sorted[prefix_len..]);
    set.push(e);
    oracle.independent(&set)
}

/// Minimum-pi element of the fundamental circuit of `X + e` in `oracle`,
/// excluding `e`. Ties break toward the smaller index.
///
/// Returns `None` when `X + e` is independent or when `e` is a loop.
/// The search is binary: `X + e` minus a prefix of the increasing-pi order
/// (index tie-break) is independent iff the circuit hits that prefix.
pub fn min_pi_circuit_element<O: IndependenceOracle>(
    oracle: &O,
    x: &[usize],
    e: usize,
    pi: &[f64],
) -> Option<usize> {
    let filtered: Vec<usize> = x.iter().copied().filter(|&f| f != e).collect();
    let mut with = filtered.clone();
    with.push(e);
    if oracle.independent(&with) {
        return None;
    }
    circuit_element_assuming_dependent(oracle, &filtered, e, pi)
}

fn circuit_element_assuming_dependent<O: IndependenceOracle>(
    oracle: &O,
    x_without_e: &[usize],
    e: usize,
    pi: &[f64],
) -> Option<usize> {
    if x_without_e.is_empty() {
        return None;
    }
    let mut sorted = x_without_e.to_vec();
    sorted.sort_by(|&a, &b| pi[a].partial_cmp(&pi[b]).unwrap().then(a.cmp(&b)));
    let m = sorted.len();
    // Smallest k in 1..=m such that dropping the prefix leaves an independent set.
    // hi = m + 1 means no such k has been found.
    let mut lo = 1usize;
    let mut hi = m + 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if mid <= m && drop_prefix_independent(oracle, &sorted, mid, e) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo > m {
        None
    } else {
        Some(sorted[lo - 1])
    }
}

/// Price p_X(e) together with the M2 exchange element, if any.
/// `x` must be the current common set and must not contain `e`.
fn price_m2<O: IndependenceOracle>(
    oracle: &O,
    x: &[usize],
    e: usize,
    pi: &[f64],
) -> (f64, Option<usize>, bool) {
    let mut with = x.to_vec();
    with.push(e);
    if oracle.independent(&with) {
        return (0.0, None, true);
    }
    match circuit_element_assuming_dependent(oracle, x, e, pi) {
        Some(f) => (pi[f], Some(f), false),
        None => (f64::INFINITY, None, false),
    }
}

fn q_maximum_base<O: IndependenceOracle>(
    x_star: &[usize],
    q: &[f64],
    m1: &O,
    n: usize,
) -> Vec<usize> {
    let mut in_base = vec![false; n];
    let mut base = x_star.to_vec();
    for &x in x_star {
        in_base[x] = true;
    }
    let mut cand: Vec<usize> = (0..n).filter(|&e| !in_base[e]).collect();
    cand.sort_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap().then(a.cmp(&b)));
    for e in cand {
        base.push(e);
        if m1.independent(&base) {
            in_base[e] = true;
        } else {
            base.pop();
        }
    }
    base
}

fn elems_of(in_x: &[bool]) -> Vec<usize> {
    (0..in_x.len()).filter(|&e| in_x[e]).collect()
}

fn weight_of(in_x: &[bool], weights: &[f64]) -> f64 {
    let mut w = 0.0;
    for e in 0..weights.len() {
        if in_x[e] {
            w += weights[e];
        }
    }
    w
}

/// Run Algorithm 1. `internal_eps` must lie in (0, 0.1].
pub fn auction<M1, M2>(
    weights: &[f64],
    m1: &M1,
    m2: &M2,
    internal_eps: f64,
) -> Result<Solution, AuctionError>
where
    M1: IndependenceOracle,
    M2: IndependenceOracle,
{
    if !(internal_eps.is_finite() && internal_eps > 0.0 && internal_eps <= 0.1) {
        return Err(AuctionError::EpsilonOutOfRange);
    }
    let n = weights.len();
    if n == 0 {
        return Ok(Solution {
            set: Vec::new(),
            weight: 0.0,
            pi: Vec::new(),
            kappa: Vec::new(),
            phase: 0,
        });
    }
    for &w in weights {
        assert!(w.is_finite() && w > 0.0, "weights must be finite and positive");
    }
    let eps = internal_eps;
    let w_max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let big_w = 2.0 * w_max;
    let phases = num_phases(n, eps);

    let mut pi = vec![0.0; n];
    let mut kappa = vec![0.0; n];
    let mut in_x = vec![false; n];

    let mut best_set: Vec<usize> = Vec::new();
    let mut best_weight = 0.0;
    let mut best_pi = pi.clone();
    let mut best_kappa = kappa.clone();
    let mut best_phase = 0usize;

    for _phase in 1..=phases {
        let in_x_star = in_x.clone();
        let x_star = elems_of(&in_x_star);
        let mut in_y = in_x_star.clone();

        let mut q_star = vec![0.0; n];
        for e in 0..n {
            if in_x_star[e] {
                q_star[e] = kappa[e];
            } else {
                let x_now = elems_of(&in_x_star);
                let (p, _, _) = price_m2(m2, &x_now, e, &pi);
                let u = (weights[e] - p).max(0.0);
                q_star[e] = ceil_lambda(u, eps, big_w);
            }
        }

        let base = q_maximum_base(&x_star, &q_star, m1, n);

        for e in 0..n {
            if in_x_star[e] {
                continue;
            }
            let lambda = q_star[e];
            let x_now = elems_of(&in_x);
            let (p, circuit_f, m2_indep) = price_m2(m2, &x_now, e, &pi);
            let u = weights[e] - p;

            let mut test = Vec::new();
            for &b in &base {
                if q_star[b] > lambda {
                    test.push(b);
                }
            }
            for y in 0..n {
                if in_y[y] && q_star[y] == lambda {
                    test.push(y);
                }
            }
            test.push(e);
            let m1_ok = m1.independent(&test);

            if m1_ok && u > 0.0 && u + eps * weights[e] >= lambda {
                in_y[e] = true;
                if m2_indep {
                    debug_assert!(!in_x[e]);
                    in_x[e] = true;
                } else {
                    let f = circuit_f.expect("M2 circuit element missing");
                    debug_assert!(in_x[f] && f != e);
                    in_x[f] = false;
                    in_x[e] = true;
                }
                pi[e] = p + eps * weights[e];
                kappa[e] = u + eps * weights[e];
            }
        }

        let weight = weight_of(&in_x, weights);
        if weight > best_weight {
            best_weight = weight;
            best_set = elems_of(&in_x);
            best_pi = pi.clone();
            best_kappa = kappa.clone();
            best_phase = _phase;
        }
    }

    Ok(Solution {
        set: best_set,
        weight: best_weight,
        pi: best_pi,
        kappa: best_kappa,
        phase: best_phase,
    })
}

/// Theorem 1: Algorithm 1 with internal epsilon `eps / 5`.
/// Caller epsilon above 1/2 is rejected.
pub fn approximate<M1, M2>(
    weights: &[f64],
    m1: &M1,
    m2: &M2,
    eps: f64,
) -> Result<Solution, AuctionError>
where
    M1: IndependenceOracle,
    M2: IndependenceOracle,
{
    if !(eps.is_finite() && eps > 0.0 && eps <= 0.5) {
        return Err(AuctionError::EpsilonOutOfRange);
    }
    auction(weights, m1, m2, eps / 5.0)
}
