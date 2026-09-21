use crate::error::LoscError;

/// Binary pooling matrix Φ ∈ {0,1}^{m×n}, stored row-major as `u8` bits.
///
/// Indices are 0-based (`0..m` rows, `0..n` columns). A nonzero entry is
/// treated as membership (`Φ_ij = 1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolingMatrix {
    m: usize,
    n: usize,
    data: Vec<u8>,
}

impl PoolingMatrix {
    /// All-zero `m × n` matrix.
    pub fn zeros(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            data: vec![0; m.saturating_mul(n)],
        }
    }

    /// Identity `n × n` pooling (one sample per pool).
    pub fn identity(n: usize) -> Self {
        let mut phi = Self::zeros(n, n);
        for i in 0..n {
            phi.data[i * n + i] = 1;
        }
        phi
    }

    /// Build from row vectors of bits. Empty `rows` yields a `0 × 0` matrix.
    pub fn from_rows(rows: &[Vec<u8>]) -> Result<Self, LoscError> {
        let m = rows.len();
        if m == 0 {
            return Ok(Self::zeros(0, 0));
        }
        let n = rows[0].len();
        let mut data = Vec::with_capacity(m * n);
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(LoscError::JaggedRows {
                    row: i,
                    expected: n,
                    got: row.len(),
                });
            }
            for &v in row {
                data.push(if v == 0 { 0 } else { 1 });
            }
        }
        Ok(Self { m, n, data })
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `Φ_ij` as 0 or 1.
    pub fn get(&self, i: usize, j: usize) -> u8 {
        self.data[i * self.n + j]
    }

    /// Set `Φ_ij`; any nonzero `value` is stored as 1.
    pub fn set(&mut self, i: usize, j: usize, value: u8) {
        self.data[i * self.n + j] = if value == 0 { 0 } else { 1 };
    }

    pub fn row(&self, i: usize) -> &[u8] {
        let start = i * self.n;
        &self.data[start..start + self.n]
    }

    /// Number of ones in row `i`.
    pub fn row_sum(&self, i: usize) -> usize {
        self.row(i).iter().map(|&v| v as usize).sum()
    }

    /// Number of ones in column `j`.
    pub fn col_sum(&self, j: usize) -> usize {
        (0..self.m)
            .map(|i| self.data[i * self.n + j] as usize)
            .sum()
    }
}
