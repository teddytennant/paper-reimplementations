//! Temporal dominance operator `Φ_T` (Equation 4) and Theorem 4.1.

use crate::oracle::ConflictOracle;
use crate::types::Retrieved;

/// Temporal dominance (Eq. 4):
///
/// ```text
/// Φ_T(R_k) = { c_i ∈ R_k | ∄ c_j ∈ R_k : κ(c_i, c_j)=1 ∧ τ_j > τ_i }
/// ```
///
/// Pairwise κ is evaluated over `i ≠ j`, so the oracle is called at most
/// `|R_k|(|R_k|-1)` times (strictly `O(|R_k|^2)` as in §4.2).
///
/// Theorem 4.1: if at least one valid chunk `S_new` is in `R_k`, every older
/// κ-contradictory `S_old` is excised. Non-contradicting history is kept.
pub fn temporal_dominance<C: Clone>(
    rk: &[Retrieved<C>],
    oracle: &impl ConflictOracle<C>,
    query: &str,
) -> Vec<Retrieved<C>> {
    let n = rk.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if oracle.contradicts(&rk[i].chunk.content, &rk[j].chunk.content, query)
                && rk[j].chunk.timestamp > rk[i].chunk.timestamp
            {
                keep[i] = false;
                break;
            }
        }
    }
    rk.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, r)| r.clone())
        .collect()
}
