//! Zonotope compression for represented generator multisets.
//!
//! Positioning follows the alpha = 2 construction in the proof of Lemma 3.4:
//! ell_1 Lewis weights, then A = sqrt(2) * M^{-1/2}. That map puts the
//! zonotope between the Euclidean unit ball and the ball of radius 2 sqrt(r).
//! Lemma 3.4's displayed sqrt(2r) ball is not what that calculation proves.
//! Delta uses the proof of Lemma 3.5, delta = c_r * eta / (8 * sqrt(r)),
//! which is the loss bound for aspect ratio 2 sqrt(r). It is not replaced
//! by a "corrected" formula.
//!
//! The linear isomorphism from the generator span to R^r is the right singular
//! basis of the generator matrix (eigenvectors of G^T G). The paper allows
//! any linear isomorphism.
//!
//! Net points a / ||a||_2 for nonzero integer a with ||a||_infinity <= L+1
//! are deduplicated when their Euclidean distance is at most 1e-12. Scalar
//! multiples land on the same direction and collapse under that rule.

use crate::linalg::{
    apply_v, apply_vt, invert_spd, l2, mat_vec, numerical_right_basis, quad_form,
    signed_sqrt_inverse, sphere_constant, zeros,
};

/// Iteration count from the proof of Lemma 3.4.
/// T = ceil(2 * ln(2 * ln(m+1) / ln(2))) for m >= 2.
pub(crate) fn lewis_iteration_count(m: usize) -> usize {
    if m < 2 {
        return 0;
    }
    let inner = 2.0 * ((m as f64) + 1.0).ln() / 2.0_f64.ln();
    let t = 2.0 * inner.ln();
    t.ceil() as usize
}

/// Exact ell_1 Lewis-weight iteration. No damping.
/// m = 1 returns [1]. Otherwise start at w_i = 1 and apply the exact update
/// `t` times: w_i = sqrt(g_i^T M^{-1} g_i), M = sum_j g_j g_j^T / w_j.
pub(crate) fn lewis_weights(rows: &[Vec<f64>], iterations: usize) -> Vec<f64> {
    let m = rows.len();
    assert!(m >= 1, "Lewis weights need at least one row");
    if m == 1 {
        return vec![1.0];
    }
    let r = rows[0].len();
    let mut w = vec![1.0; m];
    for _ in 0..iterations {
        let mut gram = zeros(r, r);
        for (row, &wi) in rows.iter().zip(w.iter()) {
            let scale = 1.0 / wi;
            for a in 0..r {
                for b in 0..r {
                    gram[a][b] += scale * row[a] * row[b];
                }
            }
        }
        let inv = invert_spd(&gram);
        for i in 0..m {
            let quad = quad_form(&inv, &rows[i]);
            // A negative quadratic form here is roundoff. A zero weight would
            // make the next Gram matrix undefined, so the weight is floored.
            if quad <= 0.0 {
                w[i] = 1e-30;
            } else {
                w[i] = quad.sqrt();
            }
        }
    }
    w
}

/// Compress a represented zonotope so that Z_hat is contained in Z and
/// Z is contained in (1 + eta) Z_hat, up to floating-point error.
/// `eta` must lie in (0, 1). Generators that are exactly zero are dropped.
/// If none remain, the result is empty and has support 0.
pub fn compress(generators: &[Vec<f64>], eta: f64) -> Vec<Vec<f64>> {
    if !(eta > 0.0 && eta < 1.0) {
        panic!("eta must lie in (0, 1)");
    }
    if generators.is_empty() {
        return Vec::new();
    }
    let k = generators[0].len();
    if k == 0 {
        panic!("generators must be nonempty vectors");
    }
    let mut kept: Vec<Vec<f64>> = Vec::new();
    for g in generators {
        if g.len() != k {
            panic!("generators must share a dimension");
        }
        if g.iter().any(|x| !x.is_finite()) {
            panic!("generators must be finite");
        }
        if g.iter().any(|x| *x != 0.0) {
            kept.push(g.clone());
        }
    }
    if kept.is_empty() {
        return Vec::new();
    }
    let basis = numerical_right_basis(&kept);
    let r = basis.len();
    if r == 0 {
        return Vec::new();
    }
    let coords: Vec<Vec<f64>> = kept
        .iter()
        .map(|g| apply_vt(&basis, g))
        .filter(|c| c.iter().any(|x| *x != 0.0))
        .collect();
    if coords.is_empty() {
        return Vec::new();
    }
    let m = coords.len();
    let t = lewis_iteration_count(m);
    let weights = lewis_weights(&coords, t);
    let mut m_tilde = zeros(r, r);
    for (row, &wi) in coords.iter().zip(weights.iter()) {
        let scale = 1.0 / wi;
        for a in 0..r {
            for b in 0..r {
                m_tilde[a][b] += scale * row[a] * row[b];
            }
        }
    }
    // A = sqrt(2) * M^{-1/2}. The paired square root is A^{-1} = M^{1/2} / sqrt(2).
    let (a_map, a_inv, _ridged) = signed_sqrt_inverse(&m_tilde, 2.0_f64.sqrt());
    let mut ys: Vec<Vec<f64>> = coords.iter().map(|g| mat_vec(&a_map, g)).collect();
    for y in &mut ys {
        if y.iter().any(|v| !v.is_finite()) {
            panic!("positioning map produced a non-finite coordinate");
        }
    }
    let c_r = sphere_constant(r);
    let delta = c_r * eta / (8.0 * (r as f64).sqrt());
    if !(delta > 0.0 && delta.is_finite()) {
        panic!("net spacing delta is not positive");
    }
    let l_delta = ((r as f64).sqrt() / delta).ceil() + 1.0;
    if !l_delta.is_finite() || l_delta > (i64::MAX as f64) / 4.0 {
        panic!("lattice net bound is too large for this eta and rank");
    }
    let l_delta_i = l_delta as i64;
    let grouped = aggregate_by_net(&ys, l_delta_i);
    let mut out = Vec::with_capacity(grouped.len());
    for sum_y in grouped {
        let pulled = mat_vec(&a_inv, &sum_y);
        let g = apply_v(&basis, &pulled);
        if g.iter().any(|x| *x != 0.0) {
            out.push(g);
        }
    }
    out
}

/// Group y_i by nearest net point. Points within 1e-12 are one group.
fn aggregate_by_net(ys: &[Vec<f64>], l_delta: i64) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut sums: Vec<Vec<f64>> = Vec::new();
    for y in ys {
        let norm = l2(y);
        if norm == 0.0 {
            continue;
        }
        let dir: Vec<f64> = y.iter().map(|v| v / norm).collect();
        let net = nearest_net_point(&dir, l_delta);
        let mut found = None;
        for (j, p) in points.iter().enumerate() {
            if dist_sq(p, &net) <= 1e-24 {
                found = Some(j);
                break;
            }
        }
        if let Some(j) = found {
            for (s, v) in sums[j].iter_mut().zip(y.iter()) {
                *s += *v;
            }
        } else {
            points.push(net);
            sums.push(y.clone());
        }
    }
    sums
}

fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Nearest point of the lattice net to a unit vector.
/// L = l_delta, and the cube is ||a||_infinity <= L + 1.
/// Small nets are scanned exhaustively. Larger nets use a cone search that
/// contains every lattice direction within the distance already achieved by
/// the paper's rounding construction, so the argmin matches the exhaustive scan.
pub(crate) fn nearest_net_point(unit: &[f64], l_delta: i64) -> Vec<f64> {
    let r = unit.len();
    assert!(r >= 1, "net dimension must be positive");
    assert!(l_delta >= 1, "lattice parameter L must be at least 1");
    let n_inf = l_delta + 1;
    let unorm = l2(unit);
    assert!(unorm > 0.0, "net query must be nonzero");
    let u: Vec<f64> = unit.iter().map(|x| x / unorm).collect();
    let mut best_cos = f64::NEG_INFINITY;
    let mut best_a = vec![0i64; r];
    consider_rounded(&u, l_delta, n_inf, &mut best_cos, &mut best_a);
    // A few short vectors, in case rounding underflows on a tiny L.
    for k in 0..r {
        let mut a = vec![0i64; r];
        a[k] = 1;
        consider(&a, &u, &mut best_cos, &mut best_a);
        a[k] = -1;
        consider(&a, &u, &mut best_cos, &mut best_a);
    }
    let use_brute = lattice_count(r, n_inf) <= 80_000;
    if use_brute {
        brute_best(&u, n_inf, &mut best_cos, &mut best_a);
    } else {
        cone_best(&u, n_inf, &mut best_cos, &mut best_a);
    }
    normalize_lattice(&best_a)
}

fn lattice_count(r: usize, n_inf: i64) -> u128 {
    let side = (2 * n_inf + 1) as u128;
    let mut count = 1u128;
    for _ in 0..r {
        if count > u128::MAX / side {
            return u128::MAX;
        }
        count *= side;
    }
    count
}

fn consider_rounded(u: &[f64], l_delta: i64, n_inf: i64, best_cos: &mut f64, best_a: &mut Vec<i64>) {
    let mut a = vec![0i64; u.len()];
    for k in 0..u.len() {
        let rounded = (l_delta as f64 * u[k]).round();
        let mut ak = if rounded.is_finite() {
            rounded as i64
        } else if u[k] >= 0.0 {
            n_inf
        } else {
            -n_inf
        };
        if ak > n_inf {
            ak = n_inf;
        }
        if ak < -n_inf {
            ak = -n_inf;
        }
        a[k] = ak;
    }
    consider(&a, u, best_cos, best_a);
}

fn brute_best(u: &[f64], n_inf: i64, best_cos: &mut f64, best_a: &mut Vec<i64>) {
    let r = u.len();
    let mut a = vec![-n_inf; r];
    loop {
        consider(&a, u, best_cos, best_a);
        let mut k = 0;
        loop {
            if k == r {
                return;
            }
            a[k] += 1;
            if a[k] <= n_inf {
                break;
            }
            a[k] = -n_inf;
            k += 1;
        }
    }
}

/// Enumerate a superset of lattice vectors whose direction is at least as
/// close to `u` as the incumbent. The incumbent includes the rounding
/// construction, which lies at distance less than delta, so the true nearest
/// net point is inside this cone.
fn cone_best(u: &[f64], n_inf: i64, best_cos: &mut f64, best_a: &mut Vec<i64>) {
    let r = u.len();
    let d0 = (2.0 * (1.0 - *best_cos).max(0.0)).sqrt();
    let d = d0 + 1e-7;
    let k_star = (0..r)
        .max_by(|&i, &j| {
            u[i]
                .abs()
                .partial_cmp(&u[j].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        })
        .unwrap();
    let u_abs = u[k_star].abs();
    if !(d < 0.9 * u_abs) {
        // The cone bound needs a positive gap. This should not happen for the
        // delta used by compression. Fall back only when the cube is modest.
        if lattice_count(r, n_inf) <= 5_000_000 {
            brute_best(u, n_inf, best_cos, best_a);
        }
        return;
    }
    let sign = if u[k_star] >= 0.0 { 1i64 } else { -1i64 };
    for s_abs in 1..=n_inf {
        let s = sign * s_abs;
        let rho_lo = (s_abs as f64) / (u_abs + d);
        let rho_hi = (s_abs as f64) / (u_abs - d);
        let mut ranges = Vec::with_capacity(r);
        let mut empty = false;
        for j in 0..r {
            if j == k_star {
                ranges.push((s, s));
                continue;
            }
            let (lo, hi) = axis_bounds(u[j], d, rho_lo, rho_hi);
            let mut lo_i = (lo - 2.0).floor() as i64;
            let mut hi_i = (hi + 2.0).ceil() as i64;
            if lo_i < -n_inf {
                lo_i = -n_inf;
            }
            if hi_i > n_inf {
                hi_i = n_inf;
            }
            if lo_i > hi_i {
                empty = true;
                break;
            }
            ranges.push((lo_i, hi_i));
        }
        if empty {
            continue;
        }
        enumerate_ranges(&ranges, &mut |a| consider(a, u, best_cos, best_a));
    }
}

fn axis_bounds(u_j: f64, d: f64, rho_lo: f64, rho_hi: f64) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &rho in &[rho_lo, rho_hi] {
        let lower = rho * (u_j - d);
        let upper = rho * (u_j + d);
        if lower < lo {
            lo = lower;
        }
        if upper > hi {
            hi = upper;
        }
    }
    (lo, hi)
}

fn enumerate_ranges(ranges: &[(i64, i64)], visit: &mut dyn FnMut(&[i64])) {
    let r = ranges.len();
    let mut a: Vec<i64> = ranges.iter().map(|(lo, _)| *lo).collect();
    loop {
        visit(&a);
        let mut k = 0;
        loop {
            if k == r {
                return;
            }
            a[k] += 1;
            if a[k] <= ranges[k].1 {
                break;
            }
            a[k] = ranges[k].0;
            k += 1;
        }
    }
}

fn consider(a: &[i64], u: &[f64], best_cos: &mut f64, best_a: &mut Vec<i64>) {
    if a.iter().all(|&x| x == 0) {
        return;
    }
    let mut dot_v = 0.0;
    let mut norm_sq = 0.0;
    for j in 0..a.len() {
        let aj = a[j] as f64;
        dot_v += u[j] * aj;
        norm_sq += aj * aj;
    }
    let cos = dot_v / norm_sq.sqrt();
    let better = if cos > *best_cos + 1e-18 {
        true
    } else if (*best_cos - cos).abs() <= 1e-18 {
        a < best_a.as_slice()
    } else {
        false
    };
    if better {
        *best_cos = cos;
        best_a.clone_from_slice(a);
    }
}

fn normalize_lattice(a: &[i64]) -> Vec<f64> {
    let mut norm_sq = 0.0;
    for &ai in a {
        let x = ai as f64;
        norm_sq += x * x;
    }
    let norm = norm_sq.sqrt();
    a.iter().map(|&ai| (ai as f64) / norm).collect()
}

/// Cosine of the best net point. Test helper for comparing searches.
#[cfg(test)]
pub(crate) fn best_net_cosine_brute(unit: &[f64], l_delta: i64) -> f64 {
    let n_inf = l_delta + 1;
    let unorm = l2(unit);
    let u: Vec<f64> = unit.iter().map(|x| x / unorm).collect();
    let mut best_cos = f64::NEG_INFINITY;
    let mut best_a = vec![0i64; u.len()];
    brute_best(&u, n_inf, &mut best_cos, &mut best_a);
    best_cos
}

#[cfg(test)]
pub(crate) fn best_net_cosine_cone(unit: &[f64], l_delta: i64) -> f64 {
    let n_inf = l_delta + 1;
    let unorm = l2(unit);
    let u: Vec<f64> = unit.iter().map(|x| x / unorm).collect();
    let mut best_cos = f64::NEG_INFINITY;
    let mut best_a = vec![0i64; u.len()];
    consider_rounded(&u, l_delta, n_inf, &mut best_cos, &mut best_a);
    cone_best(&u, n_inf, &mut best_cos, &mut best_a);
    best_cos
}

/// Exposed so tests can check the fixed-point identity on a chosen matrix.
#[cfg(test)]
pub(crate) fn lewis_gram_quads(rows: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let r = rows[0].len();
    let mut gram = zeros(r, r);
    for (row, &wi) in rows.iter().zip(weights) {
        let scale = 1.0 / wi;
        for a in 0..r {
            for b in 0..r {
                gram[a][b] += scale * row[a] * row[b];
            }
        }
    }
    let inv = invert_spd(&gram);
    rows.iter().map(|row| quad_form(&inv, row)).collect()
}

