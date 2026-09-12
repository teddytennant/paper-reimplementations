//! Algorithm 1 / Theorem 17: derivative-based polynomial recovery.

use crate::field::{self, as_exponent};
use crate::poly::{SparsePoly, Term, UniPoly};

/// Recover a y-separated sparse polynomial from its evaluation at b and
/// the evaluations of its partials ∂f/∂x_k at b (Algorithm 1).
///
/// e_ik = (B_{i,k} / A_i) * b_k, then c_i = A_i * (∏ b_j^{e_ij})^{-1}.
pub fn recover(
    b: &[u64],
    f_b: &UniPoly,
    partials_b: &[UniPoly],
    max_deg: u32,
) -> Result<SparsePoly, String> {
    let n = b.len();
    if partials_b.len() != n {
        return Err("need one partial per variable".into());
    }
    for &bk in b {
        if bk == 0 {
            return Err("evaluation point must lie in (F_p*)^n".into());
        }
    }
    let degrees = f_b.support_degrees();
    let mut terms = Vec::new();
    for di in degrees {
        let ai = f_b.coeff(di);
        if ai == 0 {
            continue;
        }
        let ai_inv = field::inv(ai);
        let mut xexp = vec![0u32; n];
        for k in 0..n {
            let bik = partials_b[k].coeff(di);
            // e_ik = (B_{i,k} / A_i) * b_k
            let e_field = field::mul(field::mul(bik, ai_inv), b[k]);
            xexp[k] = as_exponent(e_field, max_deg)?;
        }
        // c_i = A_i * (∏ b_j^{e_ij})^{-1}
        let mut prod = 1u64;
        for k in 0..n {
            prod = field::mul(prod, field::pow(b[k], xexp[k] as u64));
        }
        let ci = field::mul(ai, field::inv(prod));
        terms.push(Term {
            xexp,
            yexp: di,
            coeff: ci,
        });
    }
    Ok(SparsePoly::from_terms(n, terms))
}
