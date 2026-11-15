//! Las-Vegas wrapper around Algorithm 1.
//!
//! Proposition 1 of the paper: even an incorrect REF `L` still satisfies
//! rowspace(L) subset rowspace(A). Equality holds iff every original row
//! of `A` reduces to zero against `L`. If that check fails, fall back to
//! deterministic REF. The output is then always a correct REF of `A`.

use crate::matrix::{deterministic_ref, Matrix, RowEchelon};
use crate::pref::algorithm1;
use rand::Rng;

/// Reduce every row of `a` against `l`. True iff rowspace(A) subset rowspace(L).
pub fn all_rows_reduce_to_zero(l: &RowEchelon, a: &Matrix) -> bool {
    l.reduces_all(a)
}

/// Accept `l` if it spans the rowspace of `a`, otherwise replace it by
/// the deterministic REF of `a`.
pub fn certify_or_fallback(l: &RowEchelon, a: &Matrix) -> RowEchelon {
    if all_rows_reduce_to_zero(l, a) {
        l.clone()
    } else {
        deterministic_ref(a)
    }
}

/// Algorithm 1 followed by rowspace certification (Las Vegas).
pub fn las_vegas_ref<R: Rng>(
    a: &Matrix,
    already_echelon: usize,
    ell: usize,
    s: usize,
    rng: &mut R,
) -> RowEchelon {
    let l = algorithm1(a, already_echelon, ell, s, rng);
    certify_or_fallback(&l, a)
}

/// Las Vegas with `R = 0` and `ell = ceil(sqrt(N))`.
pub fn certified_ref<R: Rng>(a: &Matrix, s: usize, rng: &mut R) -> RowEchelon {
    let n = a.nrows();
    if n == 0 {
        return RowEchelon::empty(a.field(), a.ncols());
    }
    let ell = crate::pref::default_block_size(n).min(n);
    las_vegas_ref(a, 0, ell, s, rng)
}
