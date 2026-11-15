use std::fmt;

/// Recoverable parameter or dimension error.
#[derive(Debug, Clone, PartialEq)]
pub enum LoscError {
    /// Bernoulli parameter `p` is outside `[0, 1]`.
    InvalidProbability { p: f64 },
    /// Dilution-constrained pool size `pr` exceeds the number of samples `n`.
    PoolSizeExceedsN { pr: usize, n: usize },
    /// Sample-constrained design asked to pick pool indices from an empty set.
    EmptyPoolIndexSet { pc: usize },
    /// Vector length does not match a matrix dimension.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// Rows of a pooling matrix are not all the same length.
    JaggedRows { row: usize, expected: usize, got: usize },
}

impl fmt::Display for LoscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoscError::InvalidProbability { p } => {
                write!(f, "Bernoulli p={p} is outside [0, 1]")
            }
            LoscError::PoolSizeExceedsN { pr, n } => {
                write!(f, "pool size pr={pr} exceeds n={n}")
            }
            LoscError::EmptyPoolIndexSet { pc } => {
                write!(f, "sample-constrained design with pc={pc} but m=0")
            }
            LoscError::LengthMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} length {got} != expected {expected}"),
            LoscError::JaggedRows {
                row,
                expected,
                got,
            } => write!(
                f,
                "row {row} has length {got}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for LoscError {}
