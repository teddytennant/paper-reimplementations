//! Field GCD via Algorithms 1 and 3 (section 4.2 reduction, no integer pipeline).

use crate::factor::factor_partial;
use crate::poly::{SparsePoly, UniPoly};
use crate::recover::recover;

/// Recover G = gcd(A, B) for y-separated sparse A, B in F_p[x_1..x_n, y].
///
/// Section 4.2: G0(y) = monic gcd(A(b,y), B(b,y)). Take F = A (so G divides F),
/// get factor derivatives of G via Theorem 24, recover G with Algorithm 1.
pub fn field_gcd(a: &SparsePoly, b: &SparsePoly, eval: &[u64]) -> Result<SparsePoly, String> {
    assert_eq!(a.n, b.n);
    let n = a.n;
    assert_eq!(eval.len(), n);
    let a_b = a.eval_x(eval);
    let b_b = b.eval_x(eval);
    let g0 = UniPoly::gcd(&a_b, &b_b);
    if g0.deg() <= 0 {
        return Ok(SparsePoly::constant(n, 1));
    }
    // F = A, H0 = A(b)/G0. Need Bézout for G0 and H0: u G0 + v H0 = 1.
    let (q, r) = a_b.div_rem(&g0);
    if !r.is_zero() {
        return Err("G0 does not divide A(b)".into());
    }
    let h0 = q;
    let (g_check, u, v) = UniPoly::xgcd(&g0, &h0);
    if g_check.deg() != 0 {
        return Err("G0 and H0 are not coprime at this evaluation".into());
    }
    // xgcd returns monic gcd; scale u,v so u G0 + v H0 = 1 (already, gcd monic of deg 0 is 1).
    let mut partials = Vec::with_capacity(n);
    let max_deg = a.total_degree().max(b.total_degree());
    for k in 0..n {
        let df = a.partial_x(k).eval_x(eval);
        let (dg, _dh) = factor_partial(&df, &g0, &h0, &u, &v);
        partials.push(dg);
    }
    recover(eval, &g0, &partials, max_deg)
}
