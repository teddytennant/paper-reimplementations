//! Dense matrices over GF(p) and incremental row echelon form.

use crate::field::Fp;
use rand::Rng;
use std::fmt;

/// Dense row-major matrix with entries in `0..p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    field: Fp,
    data: Vec<i64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize, field: Fp) -> Self {
        Matrix {
            rows,
            cols,
            field,
            data: vec![0; rows.saturating_mul(cols)],
        }
    }

    pub fn identity(n: usize, field: Fp) -> Self {
        let mut m = Matrix::zeros(n, n, field);
        for i in 0..n {
            m.set(i, i, 1);
        }
        m
    }

    /// Build from integer rows, reducing each entry into the field.
    pub fn from_i64_rows(field: Fp, rows: &[&[i64]]) -> Self {
        let r = rows.len();
        let c = if r == 0 { 0 } else { rows[0].len() };
        let mut m = Matrix::zeros(r, c, field);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), c, "ragged row");
            for (j, &v) in row.iter().enumerate() {
                m.set(i, j, field.reduce(v));
            }
        }
        m
    }

    pub fn random<R: Rng>(rows: usize, cols: usize, field: Fp, rng: &mut R) -> Self {
        let p = field.modulus();
        let data = (0..rows.saturating_mul(cols))
            .map(|_| rng.gen_range(0..p))
            .collect();
        Matrix {
            rows,
            cols,
            field,
            data,
        }
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }
    pub fn field(&self) -> Fp {
        self.field
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> i64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: i64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<i64> {
        if self.cols == 0 {
            return Vec::new();
        }
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn rows_vec(&self) -> Vec<Vec<i64>> {
        (0..self.rows).map(|i| self.row(i)).collect()
    }

    /// Stack `self` on top of `other` (same field and column count).
    pub fn vstack(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.field, other.field);
        assert_eq!(self.cols, other.cols);
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Matrix {
            rows: self.rows + other.rows,
            cols: self.cols,
            field: self.field,
            data,
        }
    }

    pub fn is_zero_row(field: Fp, v: &[i64]) -> bool {
        v.iter().all(|&x| x == 0) || field.modulus() == 0
    }

    /// Rank via deterministic incremental REF.
    pub fn rank(&self) -> usize {
        deterministic_ref(self).rank()
    }

    /// True iff every row of `other` lies in the rowspace of `self`.
    pub fn rowspace_contains(&self, other: &Matrix) -> bool {
        assert_eq!(self.field, other.field);
        assert_eq!(self.cols, other.cols);
        let l = deterministic_ref(self);
        l.reduces_all(other)
    }

    /// True iff the two matrices have the same rowspace.
    pub fn rowspace_eq(&self, other: &Matrix) -> bool {
        self.rowspace_contains(other) && other.rowspace_contains(self)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}x{} over {}", self.rows, self.cols, self.field)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Incremental row echelon form: pivot 1s, zeros below each pivot,
/// strictly increasing pivot columns.
#[derive(Clone, Debug)]
pub struct RowEchelon {
    field: Fp,
    cols: usize,
    rows: Vec<Vec<i64>>,
    pivots: Vec<usize>,
}

impl RowEchelon {
    pub fn empty(field: Fp, cols: usize) -> Self {
        RowEchelon {
            field,
            cols,
            rows: Vec::new(),
            pivots: Vec::new(),
        }
    }

    /// Load already-echelon rows (paper: A1). Zero rows are dropped.
    /// Each nonzero row is scaled so its leading entry is 1.
    pub fn from_prefix(field: Fp, prefix: &[Vec<i64>]) -> Self {
        let cols = prefix.first().map(|r| r.len()).unwrap_or(0);
        let mut e = RowEchelon::empty(field, cols);
        for row in prefix {
            if cols == 0 {
                continue;
            }
            assert_eq!(row.len(), cols);
            let mut v = row.clone();
            if let Some(p) = first_nonzero(&v) {
                let inv = field.inv(v[p]);
                for x in v.iter_mut() {
                    *x = field.mul(*x, inv);
                }
                e.rows.push(v);
                e.pivots.push(p);
            }
        }
        e
    }

    pub fn field(&self) -> Fp {
        self.field
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }
    pub fn rank(&self) -> usize {
        self.rows.len()
    }
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
    pub fn rows(&self) -> &[Vec<i64>] {
        &self.rows
    }

    /// Reduce `v` in place against the current echelon block L.
    pub fn reduce_in_place(&self, v: &mut [i64]) {
        debug_assert_eq!(v.len(), self.cols);
        let f = self.field;
        for (row, &p) in self.rows.iter().zip(self.pivots.iter()) {
            let coeff = v[p];
            if coeff != 0 {
                for j in 0..self.cols {
                    v[j] = f.sub(v[j], f.mul(coeff, row[j]));
                }
            }
        }
    }

    pub fn reduce_vec(&self, v: &[i64]) -> Vec<i64> {
        let mut w = v.to_vec();
        if self.cols > 0 {
            self.reduce_in_place(&mut w);
        }
        w
    }

    pub fn reduces_all(&self, a: &Matrix) -> bool {
        assert_eq!(a.field(), self.field);
        assert_eq!(a.ncols(), self.cols);
        for i in 0..a.nrows() {
            let r = self.reduce_vec(&a.row(i));
            if r.iter().any(|&x| x != 0) {
                return false;
            }
        }
        true
    }

    /// Append a nonzero reduced row, scaling its pivot to 1 and
    /// clearing that column in later (larger-pivot) rows so L stays REF.
    pub fn append_reduced(&mut self, mut v: Vec<i64>) {
        let f = self.field;
        let p = first_nonzero(&v).expect("append_reduced on a zero row");
        let inv = f.inv(v[p]);
        for x in v.iter_mut() {
            *x = f.mul(*x, inv);
        }
        let idx = self.pivots.partition_point(|&c| c < p);
        for i in idx..self.rows.len() {
            let coeff = self.rows[i][p];
            if coeff != 0 {
                for j in 0..self.cols {
                    self.rows[i][j] = f.sub(self.rows[i][j], f.mul(coeff, v[j]));
                }
            }
        }
        self.rows.insert(idx, v);
        self.pivots.insert(idx, p);
    }

    pub fn to_matrix(&self) -> Matrix {
        if self.rows.is_empty() {
            return Matrix::zeros(0, self.cols, self.field);
        }
        let refs: Vec<&[i64]> = self.rows.iter().map(|r| r.as_slice()).collect();
        Matrix::from_i64_rows(self.field, &refs)
    }

    pub fn is_row_echelon(&self) -> bool {
        let mut last = None;
        for (row, &p) in self.rows.iter().zip(self.pivots.iter()) {
            if first_nonzero(row) != Some(p) {
                return false;
            }
            if row[p] != 1 {
                return false;
            }
            if let Some(q) = last {
                if p <= q {
                    return false;
                }
            }
            last = Some(p);
        }
        for (i, &p) in self.pivots.iter().enumerate() {
            for (k, row) in self.rows.iter().enumerate() {
                if k > i && row[p] != 0 {
                    return false;
                }
            }
        }
        true
    }
}

pub fn first_nonzero(v: &[i64]) -> Option<usize> {
    v.iter().position(|&x| x != 0)
}

/// Deterministic REF: reduce each original row against L and append if nonzero.
pub fn deterministic_ref(a: &Matrix) -> RowEchelon {
    let mut l = RowEchelon::empty(a.field(), a.ncols());
    for i in 0..a.nrows() {
        let mut v = a.row(i);
        if a.ncols() > 0 {
            l.reduce_in_place(&mut v);
        }
        if v.iter().any(|&x| x != 0) {
            l.append_reduced(v);
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Fp;

    #[test]
    fn identity_rank() {
        let i = Matrix::identity(4, Fp::GF2);
        assert_eq!(i.rank(), 4);
        assert!(deterministic_ref(&i).is_row_echelon());
    }

    #[test]
    fn empty_and_wide() {
        let f = Fp::P32003;
        assert_eq!(Matrix::zeros(0, 0, f).rank(), 0);
        assert_eq!(Matrix::zeros(0, 5, f).rank(), 0);
        assert_eq!(Matrix::zeros(5, 0, f).rank(), 0);
        let wide = Matrix::from_i64_rows(f, &[&[1, 2, 3, 4], &[0, 1, 1, 1]]);
        assert_eq!(wide.rank(), 2);
    }
}
