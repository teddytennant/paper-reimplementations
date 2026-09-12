//! Theorem 24 / Algorithm 3: factor partial derivatives without the auxiliary z.

use crate::poly::UniPoly;

/// Given f = g h, g monic in y, and Bézout u, v with
/// u g(b) + v h(b) = 1:
///
/// ∂g/∂x_k (b,y) = rem( v * ∂f/∂x_k (b,y), g(b,y) )
/// ∂h/∂x_k (b,y) = u * ∂f/∂x_k (b,y) + quo( v * ∂f/∂x_k (b,y), g(b,y) ) * h(b,y)
pub fn factor_partial(
    df_dxk_b: &UniPoly,
    g_b: &UniPoly,
    h_b: &UniPoly,
    u: &UniPoly,
    v: &UniPoly,
) -> (UniPoly, UniPoly) {
    let v_df = v.mul(df_dxk_b);
    let (q, r) = v_df.div_rem(g_b);
    let dg = r;
    let dh = u.mul(df_dxk_b).add(&q.mul(h_b));
    (dg, dh)
}
