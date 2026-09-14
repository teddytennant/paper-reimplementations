//! Algorithm 1: Dual-Boundary Matching based Chunk Alignment.
//!
//! `Sim` is injected (closure, dense matrix, or token-overlap cosine). No LLM
//! embeddings.
//!
//! Relative position: `rp(i) = i / (n - 1)` for `n > 1`, and `0` when `n == 1`.
//! That maps the first unit to 0 and the last unit to 1.

use std::fmt;

/// Source chunk as a half-open range of source unit indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceChunk {
    pub start: usize,
    pub end: usize,
}

impl SourceChunk {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn from_frc(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn first(&self) -> usize {
        self.start
    }

    pub fn last(&self) -> usize {
        self.end - 1
    }
}

/// Dual-boundary matching parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignParams {
    /// Mix between raw Sim and the relative-position Gaussian (`λ`).
    pub lambda: f64,
    /// Gaussian width on relative-position difference (`σ`).
    pub sigma: f64,
    /// Initial candidate window size `k`.
    pub k: usize,
    /// Increment `Δk` when the monotone path is infeasible.
    pub delta_k: usize,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            sigma: 0.25,
            k: 5,
            delta_k: 5,
        }
    }
}

/// Inclusive 0-based target end index for each source chunk.
///
/// Paper output `J` is 1-based with `j_|C| = |T|`. Here `ends[i]` is `j_{i+1} - 1`
/// and the last entry is always `n_targets - 1` on success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alignment {
    pub ends: Vec<usize>,
}

impl Alignment {
    /// Target spans as half-open `[start, end)` unit ranges, one per source chunk.
    pub fn spans(&self) -> Vec<(usize, usize)> {
        let mut prev = 0usize;
        self.ends
            .iter()
            .map(|&e| {
                let s = prev;
                prev = e + 1;
                (s, e + 1)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlignError {
    EmptyTargets,
    EmptyChunk,
    Infeasible,
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlignError::EmptyTargets => write!(f, "no target units to align"),
            AlignError::EmptyChunk => write!(f, "source chunk is empty"),
            AlignError::Infeasible => write!(f, "no monotone dual-boundary path"),
        }
    }
}

impl std::error::Error for AlignError {}

/// Relative position in `[0, 1]`: `index / (n - 1)` (`0` if `n <= 1`).
pub fn relative_position(index: usize, n: usize) -> f64 {
    if n <= 1 {
        0.0
    } else {
        index as f64 / (n - 1) as f64
    }
}

/// `SimwRP(u, v) = Sim(u, v) * ((1 - λ) + λ * exp(-((rp(u) - rp(v)) / σ)^2))`.
pub fn sim_wrp(sim: f64, rp_u: f64, rp_v: f64, lambda: f64, sigma: f64) -> f64 {
    let z = (rp_u - rp_v) / sigma;
    let w = (-z * z).exp();
    sim * ((1.0 - lambda) + lambda * w)
}

/// Whitespace token-count cosine. Empty inputs score 0.
pub fn token_overlap_cosine(a: &str, b: &str) -> f64 {
    fn counts(s: &str) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = Vec::new();
        for tok in s.split_whitespace() {
            if let Some((_, c)) = v.iter_mut().find(|(t, _)| *t == tok) {
                *c += 1;
            } else {
                v.push((tok, 1));
            }
        }
        v
    }
    let ca = counts(a);
    let cb = counts(b);
    if ca.is_empty() || cb.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    for (t, na) in &ca {
        if let Some((_, nb)) = cb.iter().find(|(u, _)| u == t) {
            dot += (*na as f64) * (*nb as f64);
        }
    }
    let na: f64 = ca.iter().map(|(_, n)| (*n as f64) * (*n as f64)).sum();
    let nb: f64 = cb.iter().map(|(_, n)| (*n as f64) * (*n as f64)).sum();
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Dense `n_src x n_tgt` score table, row-major.
#[derive(Clone, Debug)]
pub struct DenseSim {
    pub n_src: usize,
    pub n_tgt: usize,
    pub scores: Vec<f64>,
}

impl DenseSim {
    pub fn get(&self, u: usize, v: usize) -> f64 {
        self.scores[u * self.n_tgt + v]
    }
}

/// Align source chunks to target units by Algorithm 1.
///
/// `sim(source_unit, target_unit)` is the pluggable `Sim`.
pub fn align_chunks<F>(
    n_source_units: usize,
    n_target_units: usize,
    chunks: &[SourceChunk],
    sim: F,
    params: &AlignParams,
) -> Result<Alignment, AlignError>
where
    F: Fn(usize, usize) -> f64,
{
    if chunks.is_empty() {
        return Ok(Alignment { ends: Vec::new() });
    }
    if n_target_units == 0 {
        return Err(AlignError::EmptyTargets);
    }
    for c in chunks {
        if c.end <= c.start {
            return Err(AlignError::EmptyChunk);
        }
    }

    let mut k = params.k.max(1).min(n_target_units);
    loop {
        match try_align(
            n_source_units,
            n_target_units,
            chunks,
            &sim,
            params,
            k,
        ) {
            Ok(ends) => return Ok(Alignment { ends }),
            Err(AlignError::Infeasible) => {
                if k >= n_target_units {
                    return Err(AlignError::Infeasible);
                }
                let step = params.delta_k.max(1);
                k = (k + step).min(n_target_units);
            }
            Err(e) => return Err(e),
        }
    }
}

fn try_align<F>(
    n_src: usize,
    n_tgt: usize,
    chunks: &[SourceChunk],
    sim: &F,
    params: &AlignParams,
    k: usize,
) -> Result<Vec<usize>, AlignError>
where
    F: Fn(usize, usize) -> f64,
{
    let c = chunks.len();
    let last_tgt = n_tgt - 1;

    // End-boundary SimwRP for every (chunk, target).
    let mut end_score = vec![vec![0.0; n_tgt]; c];
    for i in 0..c {
        let u = chunks[i].last();
        let rp_u = relative_position(u, n_src);
        for j in 0..n_tgt {
            let rp_v = relative_position(j, n_tgt);
            end_score[i][j] = sim_wrp(sim(u, j), rp_u, rp_v, params.lambda, params.sigma);
        }
    }

    // Top-k candidate ends per chunk, by end-boundary score (higher is better).
    let mut cands: Vec<Vec<usize>> = Vec::with_capacity(c);
    for i in 0..c {
        let mut idx: Vec<usize> = (0..n_tgt).collect();
        idx.sort_by(|&a, &b| {
            end_score[i][b]
                .partial_cmp(&end_score[i][a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        idx.truncate(k);
        idx.sort_unstable();
        cands.push(idx);
    }

    // Dual-boundary path scores: end term plus next-chunk start term on that candidate.
    let mut phi = vec![vec![0.0; n_tgt]; c];
    for i in 0..c {
        for &j in &cands[i] {
            let mut s = end_score[i][j];
            if i + 1 < c && j + 1 < n_tgt {
                let u = chunks[i + 1].first();
                let v = j + 1;
                s += sim_wrp(
                    sim(u, v),
                    relative_position(u, n_src),
                    relative_position(v, n_tgt),
                    params.lambda,
                    params.sigma,
                );
            }
            phi[i][j] = s;
        }
    }

    // Monotone DP over candidates. Last chunk must land on last_tgt.
    // dp[i][p] = best score assigning chunk i to cands[i][p].
    let mut dp = vec![vec![f64::NEG_INFINITY; k.max(n_tgt)]; c];
    let mut prev = vec![vec![None; k.max(n_tgt)]; c];

    for (p, &j) in cands[0].iter().enumerate() {
        dp[0][p] = phi[0][j];
    }

    for i in 1..c {
        for (p, &j) in cands[i].iter().enumerate() {
            for (q, &jp) in cands[i - 1].iter().enumerate() {
                if jp <= j && dp[i - 1][q] + phi[i][j] > dp[i][p] {
                    dp[i][p] = dp[i - 1][q] + phi[i][j];
                    prev[i][p] = Some(q);
                }
            }
        }
    }

    let last = c - 1;
    let mut best_p = None;
    let mut best = f64::NEG_INFINITY;
    for (p, &j) in cands[last].iter().enumerate() {
        if j == last_tgt && dp[last][p] > best {
            best = dp[last][p];
            best_p = Some(p);
        }
    }
    let mut p = best_p.ok_or(AlignError::Infeasible)?;
    if !best.is_finite() {
        return Err(AlignError::Infeasible);
    }

    let mut ends = vec![0usize; c];
    ends[last] = cands[last][p];
    for i in (1..c).rev() {
        let q = prev[i][p].ok_or(AlignError::Infeasible)?;
        p = q;
        ends[i - 1] = cands[i - 1][p];
    }
    Ok(ends)
}
