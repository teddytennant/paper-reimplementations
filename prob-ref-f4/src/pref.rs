//! Algorithm 1 of Demin (arXiv:2609.14433): Monagan-Pearce-Steel
//! probabilistic row echelon form, and the closed-form success
//! probability of Corollaries 1--3.

use crate::field::Fp;
use crate::matrix::{deterministic_ref, Matrix, RowEchelon};
use rand::Rng;

/// Closed form from Corollaries 1--3:
///
/// `P(rank(BA) = rank(A)) = prod_{i=s}^{r+s-1} (1 - 1/q^i)`.
///
/// The product is empty (equals 1) when `r = 0`.
pub fn success_probability(q: i64, r: usize, s: usize) -> f64 {
    assert!(q >= 2, "field size q >= 2");
    assert!(s >= 1, "paper requires s >= 1");
    product(q, s, r.saturating_add(s).saturating_sub(1))
}

/// Proposition 1 lower bound:
/// `prod_{i=s}^{ell+s-1} (1 - 1/q^i)^T`, with `T = ceil(N / ell)`.
pub fn proposition1_lower_bound(q: i64, ell: usize, s: usize, n_blocks: usize) -> f64 {
    assert!(q >= 2);
    assert!(s >= 1);
    assert!(ell >= 1);
    let one_block = product(q, s, ell.saturating_add(s).saturating_sub(1));
    one_block.powi(n_blocks as i32)
}

fn product(q: i64, from: usize, to: usize) -> f64 {
    if from > to {
        return 1.0;
    }
    let qf = q as f64;
    let mut acc = 1.0f64;
    for i in from..=to {
        let qi = qf.powf(i as f64);
        acc *= 1.0 - 1.0 / qi;
    }
    acc
}

/// Default block size from the paper: `ell = ceil(sqrt(N))`.
pub fn default_block_size(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).sqrt().ceil() as usize
    }
}

/// Algorithm 1.
///
/// `A` is written `[A1; A2]` with `A1` the first `already_echelon` rows,
/// already in row echelon form of rank `R`. Remaining rows are partitioned
/// into blocks of size `ell`. For each block, uniform random linear
/// combinations of its rows are reduced against `L`; a nonzero remainder
/// is appended as a new pivot row. The block stops after `s` reductions
/// to zero (a total count, not consecutive).
///
/// When `already_echelon = 0` this is the general case (Remark 1).
pub fn algorithm1<R: Rng>(
    a: &Matrix,
    already_echelon: usize,
    ell: usize,
    s: usize,
    rng: &mut R,
) -> RowEchelon {
    assert!(s >= 1, "paper requires s >= 1");
    assert!(already_echelon <= a.nrows());
    let f = a.field();
    let n_total = a.nrows();
    let prefix: Vec<Vec<i64>> = (0..already_echelon).map(|i| a.row(i)).collect();
    let mut l = if already_echelon == 0 {
        RowEchelon::empty(f, a.ncols())
    } else {
        RowEchelon::from_prefix(f, &prefix)
    };

    let n = n_total - already_echelon;
    if n == 0 {
        return l;
    }
    assert!(ell >= 1 && ell <= n, "need 1 <= ell <= N");

    let t_blocks = (n + ell - 1) / ell;
    for t in 0..t_blocks {
        let start = already_echelon + t * ell;
        let ell_t = (n - t * ell).min(ell);
        let mut z = 0usize;
        for _i in 0..(ell_t + s - 1) {
            let b = random_combo(f, ell_t, rng);
            let mut c = combine_rows(a, start, ell_t, &b, f);
            if a.ncols() > 0 {
                l.reduce_in_place(&mut c);
            }
            if c.iter().all(|&x| x == 0) {
                z += 1;
                if z == s {
                    break;
                }
            } else {
                l.append_reduced(c);
            }
        }
    }
    debug_assert!(l.is_row_echelon());
    l
}

fn random_combo<R: Rng>(f: Fp, len: usize, rng: &mut R) -> Vec<i64> {
    let p = f.modulus();
    (0..len).map(|_| rng.gen_range(0..p)).collect()
}

fn combine_rows(a: &Matrix, start: usize, ell_t: usize, b: &[i64], f: Fp) -> Vec<i64> {
    let cols = a.ncols();
    let mut c = vec![0i64; cols];
    for k in 0..ell_t {
        if b[k] == 0 {
            continue;
        }
        for j in 0..cols {
            let term = f.mul(b[k], a.get(start + k, j));
            c[j] = f.add(c[j], term);
        }
    }
    c
}

/// Convenience: Algorithm 1 with `R = 0` and `ell = ceil(sqrt(N))`.
pub fn probabilistic_ref<R: Rng>(a: &Matrix, s: usize, rng: &mut R) -> RowEchelon {
    let n = a.nrows();
    if n == 0 {
        return RowEchelon::empty(a.field(), a.ncols());
    }
    let ell = default_block_size(n).min(n);
    algorithm1(a, 0, ell, s, rng)
}

/// Rank of a proposed REF (number of nonzero rows).
pub fn ref_rank(l: &RowEchelon) -> usize {
    l.rank()
}

/// True iff `l` is a (possibly incomplete) REF whose rows lie in
/// the rowspace of `a`. Always true for the output of Algorithm 1
/// (Proposition 1: rowspace(L) subset rowspace(A)).
pub fn rows_in_input_rowspace(l: &RowEchelon, a: &Matrix) -> bool {
    a.rowspace_contains(&l.to_matrix())
}

/// Keep the deterministic baseline reachable from this module.
pub fn deterministic(a: &Matrix) -> RowEchelon {
    deterministic_ref(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corollary3_formula_values() {
        // prod_{i=1}^{3} (1 - 1/2^i) = (1/2)(3/4)(7/8)
        let p = success_probability(2, 3, 1);
        assert!((p - 0.5 * 0.75 * 0.875).abs() < 1e-12);
        // empty product
        assert_eq!(success_probability(5, 0, 2), 1.0);
        // Corollary 1 with r=1,s=1: 1 - 1/q
        assert!((success_probability(3, 1, 1) - (1.0 - 1.0 / 3.0)).abs() < 1e-12);
        // s=2, r=2, q=2: prod_{i=2}^{3} (1-1/2^i) = (3/4)(7/8)
        assert!((success_probability(2, 2, 2) - 0.75 * 0.875).abs() < 1e-12);
    }

    #[test]
    fn proposition1_matches_corollary3_one_block() {
        let q = 32003i64;
        let r = 7usize;
        let s = 2usize;
        let c3 = success_probability(q, r, s);
        let p1 = proposition1_lower_bound(q, r, s, 1);
        assert!((c3 - p1).abs() < 1e-12);
    }
}
