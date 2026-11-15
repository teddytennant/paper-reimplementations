//! Timestamp re-ranking and multi-factor scoring baselines (§6.1, §6.4).
//!
//! Timestamp re-rank sorts by `τ` descending and keeps top-k, destroying valid
//! old non-contradicted memory. Multi-factor
//! `α(recency)+β(relevance)+γ(usage)+δ(stability)` assigns dominance to stale
//! chunks when usage/stability grow with age, violating Theorem 4.1.

use std::cmp::Ordering;

use crate::oracle::ConflictOracle;
use crate::types::Retrieved;

/// Sort by timestamp descending and keep the `k` newest items.
pub fn timestamp_rerank<C: Clone>(rk: &[Retrieved<C>], k: usize) -> Vec<Retrieved<C>> {
    let mut items = rk.to_vec();
    items.sort_by(|a, b| {
        b.chunk
            .timestamp
            .cmp(&a.chunk.timestamp)
            .then(
                b.relevance
                    .partial_cmp(&a.relevance)
                    .unwrap_or(Ordering::Equal),
            )
            .then(a.chunk.id.cmp(&b.chunk.id))
    });
    items.truncate(k);
    items
}

/// Weights for `α(recency) + β(relevance) + γ(usage) + δ(stability)`.
#[derive(Clone, Debug)]
pub struct MultiFactorWeights {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
}

impl MultiFactorWeights {
    /// Table 4 "timestamp-only α".
    pub fn timestamp_only() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 0.0,
        }
    }

    /// Table 4 `α + β + γ` (usage). Stale chunks crash CR-Acc.
    pub fn with_usage() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 0.0,
        }
    }

    /// Table 4 full multi-factor including stability.
    pub fn full() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 1.0,
        }
    }
}

/// Recency in `(0, 1]`: newer timestamps score higher.
pub fn recency(timestamp: u64, now: u64) -> f64 {
    1.0 / (1.0 + now.saturating_sub(timestamp) as f64)
}

pub fn multi_factor_score<C>(item: &Retrieved<C>, now: u64, w: &MultiFactorWeights) -> f64 {
    w.alpha * recency(item.chunk.timestamp, now)
        + w.beta * item.relevance
        + w.gamma * item.usage
        + w.delta * item.stability
}

/// Dominance using a composite score in place of `τ_j > τ_i`.
///
/// When `γ`/`δ` grow with age, older contradictory chunks outscore `S_new`
/// and are retained, violating Theorem 4.1.
pub fn scored_dominance<C: Clone>(
    rk: &[Retrieved<C>],
    oracle: &impl ConflictOracle<C>,
    query: &str,
    now: u64,
    weights: &MultiFactorWeights,
) -> Vec<Retrieved<C>> {
    let n = rk.len();
    let scores: Vec<f64> = rk
        .iter()
        .map(|r| multi_factor_score(r, now, weights))
        .collect();
    let mut keep = vec![true; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if oracle.contradicts(&rk[i].chunk.content, &rk[j].chunk.content, query)
                && scores[j] > scores[i]
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
