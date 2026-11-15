//! Algorithm 1: the threshold algorithm (paper Section 2).

use std::collections::{HashMap, HashSet};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::graph::{canon, common_count, Edge, Vertex};

/// Result of Algorithm 1.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdResult {
    /// `T̂`. Either `Q (m/S)^3`, or the exact prefix count `t̂` if the
    /// stream ended before `Q` triangles.
    pub estimate: f64,
    /// Stopping length `S = |S_set|`.
    pub s: usize,
    /// Observed triangle count `t̂` in the prefix (may exceed `Q` on the
    /// last inserted edge).
    pub observed: u64,
}

/// Algorithm 1 (Etgar et al., Section 2).
///
/// Require: stream length `m`; triangle threshold `Q`; a (typically
/// random-order) stream of edges of `G`.
///
/// ```text
/// S_set ← ∅; adjacency A ← ∅; t̂ ← 0
/// for each arriving edge e = (u, v):
///     t̂ ← t̂ + |N_A(u) ∩ N_A(v)|
///     S_set ← S_set ∪ {e}; update A to include e
///     if t̂ ≥ Q then break
/// S ← |S_set|
/// return T̂ = Q (m / S)^3
/// ```
///
/// If the stream ends before `Q` triangles, the estimate is `t̂` (the
/// exact prefix count, which is `T` when `S = m`). An empty stream
/// (`S = 0`) returns `0` and does not divide by `S`.
///
/// The cubic formula uses `Q`, not the observed `t̂`, even when the last
/// edge overshoots so that `t̂ > Q`.
pub fn threshold_estimate(edges: &[(Vertex, Vertex)], m: usize, q: u64) -> ThresholdResult {
    let mut s_set: HashSet<Edge> = HashSet::new();
    let mut adj: HashMap<Vertex, HashSet<Vertex>> = HashMap::new();
    let mut t_hat: u64 = 0;

    for &(u0, v0) in edges {
        if u0 == v0 {
            continue;
        }
        let (u, v) = canon(u0, v0);
        t_hat += common_count(&adj, u, v);
        s_set.insert((u, v));
        adj.entry(u).or_default().insert(v);
        adj.entry(v).or_default().insert(u);
        if t_hat >= q {
            let s = s_set.len();
            // S ≥ 1: at least the edge that just closed the threshold.
            let estimate = q as f64 * (m as f64 / s as f64).powi(3);
            return ThresholdResult {
                estimate,
                s,
                observed: t_hat,
            };
        }
    }

    let s = s_set.len();
    // End of stream before Q triangles: T̂ = t̂. Covers S = 0.
    ThresholdResult {
        estimate: t_hat as f64,
        s,
        observed: t_hat,
    }
}

/// Shuffle a copy of `edges` into a uniform random order (the stream
/// model of Section 2). Stopping time is `S = min{t : N(t) ≥ Q}`.
pub fn random_order_edges<R: Rng + ?Sized>(edges: &[Edge], rng: &mut R) -> Vec<Edge> {
    let mut out = edges.to_vec();
    out.shuffle(rng);
    out
}

/// Unspecified "sufficiently large constant" `c₀` in Theorem 1.1.
/// Chosen as `1`; the paper does not give a numerical value, and this
/// choice is not claimed to be large enough for the concentration bound.
pub const THEOREM11_C0: f64 = 1.0;

/// Threshold `Q` from Theorem 1.1: `Q = c₀ · ε^{-6} · ln(4/δ)^3`
/// with [`THEOREM11_C0`].
///
/// The paper writes `log` without a base; any base is absorbed into the
/// unspecified `c₀`. Tests must not treat this constant as numerically
/// tight.
pub fn theorem11_q(eps: f64, delta: f64) -> f64 {
    theorem11_q_with_c0(eps, delta, THEOREM11_C0)
}

/// Theorem 1.1 threshold with an explicit `c₀`.
pub fn theorem11_q_with_c0(eps: f64, delta: f64, c0: f64) -> f64 {
    c0 * eps.powf(-6.0) * (4.0 / delta).ln().powi(3)
}
