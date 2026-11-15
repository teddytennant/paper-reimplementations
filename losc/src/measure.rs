use crate::error::LoscError;
use crate::matrix::PoolingMatrix;

/// Diagonal of `D` (eq. 1): `D_ii = 1 / ∑_j Φ_ij` if the row sum is positive,
/// else `0`.
pub fn dilution_matrix(phi: &PoolingMatrix) -> Vec<f64> {
    (0..phi.m())
        .map(|i| {
            let s = phi.row_sum(i);
            if s == 0 {
                0.0
            } else {
                1.0 / (s as f64)
            }
        })
        .collect()
}

/// Noise-free measurement (eqs. 1–3).
///
/// Returns pool viral loads `y` with `y_i = (D Φ)_{i,:} x` (the mean of `x_j`
/// over samples in the pool, or `0` for an empty pool) and binary outcomes
/// `ỹ_i = 1` iff `y_i ≥ lod`.
pub fn measure(
    phi: &PoolingMatrix,
    x: &[f64],
    lod: f64,
) -> Result<(Vec<f64>, Vec<u8>), LoscError> {
    if x.len() != phi.n() {
        return Err(LoscError::LengthMismatch {
            what: "x",
            expected: phi.n(),
            got: x.len(),
        });
    }
    let d = dilution_matrix(phi);
    let mut y = vec![0.0; phi.m()];
    let mut ytilde = vec![0u8; phi.m()];
    for i in 0..phi.m() {
        if d[i] == 0.0 {
            y[i] = 0.0;
        } else {
            let mut acc = 0.0;
            for j in 0..phi.n() {
                if phi.get(i, j) != 0 {
                    acc += x[j];
                }
            }
            y[i] = d[i] * acc;
        }
        ytilde[i] = if y[i] >= lod { 1 } else { 0 };
    }
    Ok((y, ytilde))
}
