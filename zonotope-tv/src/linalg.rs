//! Symmetric eigenproblems and small dense linear algebra.
//!
//! Eigenvalues come from a cyclic Jacobi method written for this crate.
//! Numerical rank of a generator Gram matrix keeps eigenvalues strictly above
//! 1e-10 times the largest eigenvalue. That cutoff is part of the f64
//! implementation, not part of the paper.
//!
//! Inverse and inverse square root use the same eigen-decomposition. A ridge
//! is added only when an eigenvalue is non-positive from roundoff: that
//! eigenvalue is replaced by 1e-14 times the largest eigenvalue, or by 1e-30
//! if the largest eigenvalue is not positive. Positive eigenvalues are left
//! unchanged.

/// Constant from the spherical average of an absolute inner product.
/// c_r = Gamma(r/2) / (sqrt(pi) * Gamma((r+1)/2)).
/// Built from Gamma(1) = 1, Gamma(1/2) = sqrt(pi), and Gamma(z+1) = z Gamma(z),
/// which yields the recurrence c_1 = 1, c_2 = 2/pi, c_{k+2} = c_k * k / (k+1).
pub(crate) fn sphere_constant(r: usize) -> f64 {
    assert!(r >= 1, "sphere constant requires positive rank");
    let mut c = if r % 2 == 1 {
        1.0
    } else {
        2.0 / std::f64::consts::PI
    };
    let mut k = if r % 2 == 1 { 1usize } else { 2usize };
    while k < r {
        c *= (k as f64) / ((k + 1) as f64);
        k += 2;
    }
    c
}

pub(crate) fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn mat_vec(rows: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    rows.iter().map(|row| dot(row, v)).collect()
}

pub(crate) fn zeros(n: usize, m: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; m]; n]
}

pub(crate) fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = zeros(n, n);
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Columns of `v` are eigenvectors. Eigenvalues are not necessarily sorted.
pub(crate) fn jacobi_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    assert!(n > 0, "Jacobi eigen-decomposition needs a nonempty matrix");
    for row in matrix {
        assert_eq!(row.len(), n, "Jacobi input must be square");
    }
    let mut a = matrix.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let s = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = s;
            a[j][i] = s;
        }
    }
    let mut v = identity(n);
    let scale0 = frobenius(&a).max(1.0);
    for _sweep in 0..80 {
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                max_off = max_off.max(a[i][j].abs());
            }
        }
        if max_off <= 1e-15 * scale0 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() <= 1e-18 * scale0 {
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[k][p];
                    let akq = a[k][q];
                    let new_kp = c * akp - s * akq;
                    let new_kq = s * akp + c * akq;
                    a[k][p] = new_kp;
                    a[p][k] = new_kp;
                    a[k][q] = new_kq;
                    a[q][k] = new_kq;
                }
                for k in 0..n {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

fn frobenius(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Sort eigenpairs by descending eigenvalue and flip each column so its
/// largest-magnitude entry is positive. Sign flips do not change Q D Q^T.
pub(crate) fn sort_eigenpairs(evals: &[f64], evecs: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = evals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        evals[j]
            .partial_cmp(&evals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    let mut sorted_evals = vec![0.0; n];
    let mut sorted = zeros(n, n);
    for (new_j, &old_j) in order.iter().enumerate() {
        sorted_evals[new_j] = evals[old_j];
        let mut pivot = 0usize;
        let mut pivot_abs = 0.0;
        for i in 0..n {
            let mag = evecs[i][old_j].abs();
            if mag > pivot_abs {
                pivot_abs = mag;
                pivot = i;
            }
        }
        let sign = if evecs[pivot][old_j] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for i in 0..n {
            sorted[i][new_j] = sign * evecs[i][old_j];
        }
    }
    (sorted_evals, sorted)
}

pub(crate) fn symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let (evals, evecs) = jacobi_eigen(matrix);
    sort_eigenpairs(&evals, &evecs)
}

/// Returns (inverse square root, square root, whether a non-positive
/// eigenvalue was replaced). `scale` multiplies the inverse square root and
/// divides the square root, so `scale = sqrt(2)` builds the positioning map.
pub(crate) fn signed_sqrt_inverse(
    matrix: &[Vec<f64>],
    scale: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, bool) {
    let n = matrix.len();
    let (evals, q) = symmetric_eigen(matrix);
    let max_ev = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut ridged = false;
    let mut inv_sqrt_ev = vec![0.0; n];
    let mut sqrt_ev = vec![0.0; n];
    for i in 0..n {
        let mut lam = evals[i];
        if lam <= 0.0 {
            ridged = true;
            lam = if max_ev > 0.0 {
                1e-14 * max_ev
            } else {
                1e-30
            };
        }
        let root = lam.sqrt();
        sqrt_ev[i] = root / scale;
        inv_sqrt_ev[i] = scale / root;
    }
    let inv_sqrt = congruential(&q, &inv_sqrt_ev);
    let sqrt_m = congruential(&q, &sqrt_ev);
    (inv_sqrt, sqrt_m, ridged)
}

/// Q diag(d) Q^T.
fn congruential(q: &[Vec<f64>], d: &[f64]) -> Vec<Vec<f64>> {
    let n = d.len();
    let mut out = zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += q[i][k] * d[k] * q[j][k];
            }
            out[i][j] = s;
        }
    }
    out
}

pub(crate) fn invert_spd(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let (evals, q) = symmetric_eigen(matrix);
    let max_ev = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut inv_ev = vec![0.0; n];
    for i in 0..n {
        let mut lam = evals[i];
        if lam <= 0.0 {
            lam = if max_ev > 0.0 {
                1e-14 * max_ev
            } else {
                1e-30
            };
        }
        inv_ev[i] = 1.0 / lam;
    }
    congruential(&q, &inv_ev)
}

pub(crate) fn quad_form(matrix: &[Vec<f64>], v: &[f64]) -> f64 {
    let mv = mat_vec(matrix, v);
    dot(&mv, v)
}

pub(crate) fn gram_matrix(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = rows[0].len();
    let mut g = zeros(k, k);
    for row in rows {
        for i in 0..k {
            for j in 0..k {
                g[i][j] += row[i] * row[j];
            }
        }
    }
    g
}

/// Right singular basis of a matrix with rows `generators`.
/// Returns V (k by r, stored as columns) for eigenvalues of G^T G above
/// 1e-10 times the largest. Empty if the numerical rank is zero.
pub(crate) fn numerical_right_basis(generators: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let gtg = gram_matrix(generators);
    let (evals, evecs) = symmetric_eigen(&gtg);
    let max_ev = evals.iter().cloned().fold(0.0_f64, f64::max);
    if max_ev <= 0.0 {
        return Vec::new();
    }
    let cutoff = 1e-10 * max_ev;
    let mut columns = Vec::new();
    for (j, &lam) in evals.iter().enumerate() {
        if lam > cutoff {
            let mut col = vec![0.0; evecs.len()];
            for i in 0..evecs.len() {
                col[i] = evecs[i][j];
            }
            columns.push(col);
        }
    }
    columns
}

/// V is a list of orthonormal columns. Returns V^T g.
pub(crate) fn apply_vt(columns: &[Vec<f64>], g: &[f64]) -> Vec<f64> {
    columns.iter().map(|col| dot(col, g)).collect()
}

/// V y, with V stored as columns.
pub(crate) fn apply_v(columns: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = columns[0].len();
    let mut out = vec![0.0; k];
    for (coeff, col) in y.iter().zip(columns) {
        for i in 0..k {
            out[i] += coeff * col[i];
        }
    }
    out
}
