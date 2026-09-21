use crate::error::LoscError;
use crate::matrix::PoolingMatrix;

/// Algorithm 1 (Sec. III-B). Indices are 0-based.
///
/// `B` is incremented over the full row support of every positive pool, even
/// for samples later removed from `C`. Remaining candidates are sorted by `B_j`
/// descending, with smaller index first on ties. If `|C| < k` then `s_hat_k`
/// is all of `C`. If `|C| ≤ 2k` then `s_hat_2k = C`, else the first `2k`.
pub fn decode(
    ytilde: &[u8],
    phi: &PoolingMatrix,
    k: usize,
) -> Result<(Vec<usize>, Vec<usize>), LoscError> {
    if ytilde.len() != phi.m() {
        return Err(LoscError::LengthMismatch {
            what: "ytilde",
            expected: phi.m(),
            got: ytilde.len(),
        });
    }
    let n = phi.n();
    let m = phi.m();
    let mut in_c = vec![true; n];
    let mut b = vec![0usize; n];

    for i in 0..m {
        if ytilde[i] == 0 {
            for j in 0..n {
                if phi.get(i, j) != 0 {
                    in_c[j] = false;
                }
            }
        } else {
            for j in 0..n {
                if phi.get(i, j) != 0 {
                    b[j] += 1;
                }
            }
        }
    }

    let mut c: Vec<usize> = (0..n).filter(|&j| in_c[j]).collect();
    c.sort_by(|&a, &b_idx| b[b_idx].cmp(&b[a]).then(a.cmp(&b_idx)));

    let s_hat_k = if c.len() < k {
        c.clone()
    } else {
        c[..k].to_vec()
    };
    let s_hat_2k = if c.len() > 2 * k {
        c[..2 * k].to_vec()
    } else {
        c
    };
    Ok((s_hat_k, s_hat_2k))
}
