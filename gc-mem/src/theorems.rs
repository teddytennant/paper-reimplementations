//! Closed forms for Theorems 3.4 (asymptotic recall decay) and 3.5 (majority
//! vote trap), plus a hypergeometric Monte Carlo sampler.

use std::collections::HashMap;
use std::hash::Hash;

use crate::types::Retrieved;

/// Theorem 3.4 / Equation (2): `E[X] = k M / (N + M)` under uniform draws
/// from a bag of `N` stale + `M` valid chunks (Assumption 3.3: `N ≫ M`).
///
/// As `N → ∞` with `k, M` fixed, `E[X] → 0`.
pub fn expected_valid_count(k: usize, n_stale: usize, m_valid: usize) -> f64 {
    let total = n_stale + m_valid;
    if total == 0 {
        return 0.0;
    }
    let k = k.min(total);
    k as f64 * m_valid as f64 / total as f64
}

/// xorshift64* for deterministic hypergeometric draws (no extra crate).
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn seed(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform index in `0..n`. Panics if `n == 0`.
    pub fn gen_index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() as u128 * n as u128 >> 64) as usize
    }
}

/// One draw of `X ~ Hypergeometric(N+M, M, k)`: number of valid chunks in a
/// uniform sample of `k` without replacement from `N` stale + `M` valid.
pub fn hypergeometric_draw(
    k: usize,
    n_stale: usize,
    m_valid: usize,
    rng: &mut XorShift64,
) -> usize {
    let mut n = n_stale;
    let mut m = m_valid;
    let k = k.min(n + m);
    let mut x = 0;
    for _ in 0..k {
        let rem = n + m;
        if rem == 0 {
            break;
        }
        let r = rng.gen_index(rem);
        if r < m {
            x += 1;
            m -= 1;
        } else {
            n -= 1;
        }
    }
    x
}

pub fn monte_carlo_mean_valid(
    k: usize,
    n_stale: usize,
    m_valid: usize,
    draws: usize,
    rng: &mut XorShift64,
) -> f64 {
    if draws == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for _ in 0..draws {
        sum += hypergeometric_draw(k, n_stale, m_valid, rng) as f64;
    }
    sum / draws as f64
}

/// Theorem 3.5: non-discriminating attention `α_i ≈ 1/k`. `P(y = S)` is the
/// fraction of retrieved chunks asserting label `S`. With `1` valid and
/// `k-1` stale, `P(y = S_old) = (k-1)/k`.
pub fn proportional_attention<C, K, F>(rk: &[Retrieved<C>], mut label: F) -> HashMap<K, f64>
where
    K: Eq + Hash,
    F: FnMut(&C) -> K,
{
    let mut mass = HashMap::new();
    let k = rk.len();
    if k == 0 {
        return mass;
    }
    let unit = 1.0 / k as f64;
    for item in rk {
        *mass.entry(label(&item.chunk.content)).or_insert(0.0) += unit;
    }
    mass
}

/// Count-based majority vote (equivalent argmax of proportional attention).
pub fn count_vote<C, K, F>(rk: &[Retrieved<C>], mut label: F) -> Option<K>
where
    K: Eq + Hash + Clone,
    F: FnMut(&C) -> K,
{
    let mut counts: HashMap<K, usize> = HashMap::new();
    for item in rk {
        *counts.entry(label(&item.chunk.content)).or_insert(0) += 1;
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)
}
