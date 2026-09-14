//! Algorithm 2: Two-Stage Dynamic Programming for Fixed-Range Chunking.
//!
//! Length of a 0-based inclusive span `s..=e`:
//! `sum(len[s..=e]) + (e - s) * delta` (one separator cost per internal gap).

use std::fmt;

/// Paper parameters: interval `[m, M]`, separator cost `delta`, max violations `K`.
#[allow(non_snake_case)]
#[derive(Clone, Debug, PartialEq)]
pub struct FrcParams {
    /// Inclusive lower bound on packed chunk length.
    pub m: f64,
    /// Inclusive upper bound on packed chunk length.
    pub M: f64,
    /// Cost added once per internal separator inside a multi-unit chunk.
    pub delta: f64,
    /// Maximum number of out-of-range chunks in Stage 2 (`K`).
    pub k: usize,
}

impl Default for FrcParams {
    fn default() -> Self {
        Self {
            m: 256.0,
            M: 512.0,
            delta: 0.0,
            k: 8,
        }
    }
}

impl FrcParams {
    pub fn l_star(&self) -> f64 {
        (self.m + self.M) / 2.0
    }
}

/// A packed contiguous span of source units.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// Inclusive start index in the original unit sequence.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
    /// `Len` of the span, including separator cost.
    pub length: f64,
    /// True if the span is outside `[m, M]` (super-long isolate or Stage 2 violation).
    pub violated: bool,
}

/// Result of packing a unit-length sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct FrcResult {
    pub chunks: Vec<Chunk>,
    /// Sum of Stage 1/2 DP costs over non-isolated blocks. Super-long isolates add 0.
    pub cost: f64,
    /// `1` if every DP block was feasible under the strict constraint, else `2`.
    pub stage: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackError {
    /// Stage 2 still had no finite `dprel[v][T]` for `v in 1..=K`.
    Infeasible {
        block_start: usize,
        block_units: usize,
    },
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::Infeasible {
                block_start,
                block_units,
            } => write!(
                f,
                "FRC infeasible for block starting at unit {block_start} ({block_units} units)"
            ),
        }
    }
}

impl std::error::Error for PackError {}

/// Span length `Len(s, e)` with 0-based inclusive endpoints.
pub fn span_len(lens: &[usize], start: usize, end_incl: usize, delta: f64) -> f64 {
    debug_assert!(start <= end_incl && end_incl < lens.len());
    let sum: f64 = lens[start..=end_incl].iter().map(|&x| x as f64).sum();
    sum + (end_incl - start) as f64 * delta
}

fn in_range(l: f64, p: &FrcParams) -> bool {
    l >= p.m && l <= p.M
}

fn violation_cost(l: f64, p: &FrcParams) -> f64 {
    if l < p.m {
        (p.m - l).powi(2)
    } else {
        (l - p.M).powi(2)
    }
}

/// Pack source unit lengths with Algorithm 2.
pub fn pack(lens: &[usize], p: &FrcParams) -> Result<FrcResult, PackError> {
    if lens.is_empty() {
        return Ok(FrcResult {
            chunks: Vec::new(),
            cost: 0.0,
            stage: 1,
        });
    }

    let mut chunks = Vec::new();
    let mut total_cost = 0.0;
    let mut stage = 1u8;
    let mut i = 0;
    while i < lens.len() {
        if (lens[i] as f64) > p.M {
            chunks.push(Chunk {
                start: i,
                end: i + 1,
                length: lens[i] as f64,
                violated: true,
            });
            i += 1;
            continue;
        }
        let block_start = i;
        while i < lens.len() && (lens[i] as f64) <= p.M {
            i += 1;
        }
        let packed = pack_block(&lens[block_start..i], p).map_err(|_| PackError::Infeasible {
            block_start,
            block_units: i - block_start,
        })?;
        if packed.stage == 2 {
            stage = 2;
        }
        total_cost += packed.cost;
        for mut c in packed.chunks {
            c.start += block_start;
            c.end += block_start;
            chunks.push(c);
        }
    }
    Ok(FrcResult {
        chunks,
        cost: total_cost,
        stage,
    })
}

struct BlockPack {
    chunks: Vec<Chunk>,
    cost: f64,
    stage: u8,
}

fn pack_block(lens: &[usize], p: &FrcParams) -> Result<BlockPack, ()> {
    let t = lens.len();
    if t == 0 {
        return Ok(BlockPack {
            chunks: Vec::new(),
            cost: 0.0,
            stage: 1,
        });
    }
    if t == 1 && (lens[0] as f64) > p.M {
        return Ok(BlockPack {
            chunks: vec![Chunk {
                start: 0,
                end: 1,
                length: lens[0] as f64,
                violated: true,
            }],
            cost: 0.0,
            stage: 1,
        });
    }

    if let Some((cost, parents)) = stage1(lens, p) {
        return Ok(BlockPack {
            chunks: reconstruct(lens, p, &parents),
            cost,
            stage: 1,
        });
    }
    let (cost, parents) = stage2(lens, p)?;
    Ok(BlockPack {
        chunks: reconstruct(lens, p, &parents),
        cost,
        stage: 2,
    })
}

/// Stage 1: every chunk in `[m, M]`. `parents[i]` is the previous cut for prefix `i`.
fn stage1(lens: &[usize], p: &FrcParams) -> Option<(f64, Vec<Option<usize>>)> {
    let t = lens.len();
    let l_star = p.l_star();
    let mut dp = vec![f64::INFINITY; t + 1];
    let mut parents = vec![None; t + 1];
    dp[0] = 0.0;
    for i in 1..=t {
        for j in (0..i).rev() {
            let l = span_len(lens, j, i - 1, p.delta);
            if in_range(l, p) {
                let cand = dp[j] + (l - l_star).powi(2);
                if cand < dp[i] {
                    dp[i] = cand;
                    parents[i] = Some(j);
                }
            }
        }
    }
    if dp[t].is_finite() {
        Some((dp[t], parents))
    } else {
        None
    }
}

/// Stage 2: at most `K` violating chunks. Returns parents for the chosen `v*`.
fn stage2(lens: &[usize], p: &FrcParams) -> Result<(f64, Vec<Option<usize>>), ()> {
    let t = lens.len();
    let k = p.k;
    if k == 0 {
        return Err(());
    }
    let l_star = p.l_star();
    // dprel[v][i] = min cost for first i units with v violations.
    let mut dprel = vec![vec![f64::INFINITY; t + 1]; k + 1];
    // parent_j[v][i] = previous cut j; parent_v[v][i] = previous violation count.
    let mut parent_j = vec![vec![None; t + 1]; k + 1];
    let mut parent_v = vec![vec![None; t + 1]; k + 1];
    for v in 0..=k {
        dprel[v][0] = 0.0;
    }

    for i in 1..=t {
        for j in (0..i).rev() {
            let l = span_len(lens, j, i - 1, p.delta);
            if in_range(l, p) {
                let add = (l - l_star).powi(2);
                for v in 0..=k {
                    let cand = dprel[v][j] + add;
                    if cand < dprel[v][i] {
                        dprel[v][i] = cand;
                        parent_j[v][i] = Some(j);
                        parent_v[v][i] = Some(v);
                    }
                }
            } else {
                let cout = violation_cost(l, p);
                for v in 1..=k {
                    let cand = dprel[v - 1][j] + cout;
                    if cand < dprel[v][i] {
                        dprel[v][i] = cand;
                        parent_j[v][i] = Some(j);
                        parent_v[v][i] = Some(v - 1);
                    }
                }
            }
        }
    }

    let mut best_v = None;
    let mut best_cost = f64::INFINITY;
    for v in 1..=k {
        if dprel[v][t] < best_cost {
            best_cost = dprel[v][t];
            best_v = Some(v);
        }
    }
    let v_star = best_v.ok_or(())?;
    if !best_cost.is_finite() {
        return Err(());
    }

    // Flatten (v, i) parents into a cut-parent array for reconstruct.
    let mut parents = vec![None; t + 1];
    let mut i = t;
    let mut v = v_star;
    while i > 0 {
        let j = parent_j[v][i].ok_or(())?;
        let pv = parent_v[v][i].ok_or(())?;
        parents[i] = Some(j);
        i = j;
        v = pv;
    }
    Ok((best_cost, parents))
}

fn reconstruct(lens: &[usize], p: &FrcParams, parents: &[Option<usize>]) -> Vec<Chunk> {
    let t = lens.len();
    let mut cuts = Vec::new();
    let mut i = t;
    while i > 0 {
        let j = parents[i].expect("missing parent in FRC backtrack");
        cuts.push((j, i));
        i = j;
    }
    cuts.reverse();
    cuts.into_iter()
        .map(|(j, i)| {
            let length = span_len(lens, j, i - 1, p.delta);
            Chunk {
                start: j,
                end: i,
                length,
                violated: !in_range(length, p),
            }
        })
        .collect()
}

/// Minimum DP cost of a partition of `lens` under the two-stage rule, or `None`.
///
/// Super-long isolation is *not* applied here; call this on a single block.
pub fn block_cost(lens: &[usize], p: &FrcParams) -> Option<f64> {
    pack_block(lens, p).ok().map(|b| b.cost)
}
