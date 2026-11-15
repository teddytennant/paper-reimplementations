use rand::seq::index::sample;
use rand::Rng;

use crate::error::LoscError;
use crate::matrix::PoolingMatrix;

/// Pooling designs of Section III-A.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolingDesign {
    /// Each `Φ_ij ~ Bern(p)` independently.
    Unconstrained { p: f64 },
    /// Each row has exactly `pr` ones, chosen uniformly without replacement.
    DilutionConstrained { pr: usize },
    /// For each column, `pc` independent draws from `{0..m}`; the column has
    /// at most `pc` ones.
    SampleConstrained { pc: usize },
}

/// Draw a pooling matrix according to `design`.
pub fn sample_pooling<R: Rng + ?Sized>(
    m: usize,
    n: usize,
    design: PoolingDesign,
    rng: &mut R,
) -> Result<PoolingMatrix, LoscError> {
    match design {
        PoolingDesign::Unconstrained { p } => unconstrained(m, n, p, rng),
        PoolingDesign::DilutionConstrained { pr } => {
            dilution_constrained(m, n, pr, rng)
        }
        PoolingDesign::SampleConstrained { pc } => {
            sample_constrained(m, n, pc, rng)
        }
    }
}

fn unconstrained<R: Rng + ?Sized>(
    m: usize,
    n: usize,
    p: f64,
    rng: &mut R,
) -> Result<PoolingMatrix, LoscError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(LoscError::InvalidProbability { p });
    }
    let mut phi = PoolingMatrix::zeros(m, n);
    if m == 0 || n == 0 {
        return Ok(phi);
    }
    for i in 0..m {
        for j in 0..n {
            if rng.gen_bool(p) {
                phi.set(i, j, 1);
            }
        }
    }
    Ok(phi)
}

fn dilution_constrained<R: Rng + ?Sized>(
    m: usize,
    n: usize,
    pr: usize,
    rng: &mut R,
) -> Result<PoolingMatrix, LoscError> {
    if pr > n {
        return Err(LoscError::PoolSizeExceedsN { pr, n });
    }
    let mut phi = PoolingMatrix::zeros(m, n);
    if pr == 0 || m == 0 || n == 0 {
        return Ok(phi);
    }
    for i in 0..m {
        let idx = sample(rng, n, pr);
        for j in idx.iter() {
            phi.set(i, j, 1);
        }
    }
    Ok(phi)
}

fn sample_constrained<R: Rng + ?Sized>(
    m: usize,
    n: usize,
    pc: usize,
    rng: &mut R,
) -> Result<PoolingMatrix, LoscError> {
    if pc > 0 && m == 0 {
        return Err(LoscError::EmptyPoolIndexSet { pc });
    }
    let mut phi = PoolingMatrix::zeros(m, n);
    if pc == 0 || m == 0 || n == 0 {
        return Ok(phi);
    }
    for j in 0..n {
        for _ in 0..pc {
            let i = rng.gen_range(0..m);
            phi.set(i, j, 1);
        }
    }
    Ok(phi)
}
