//! Dense GF(p) probabilistic row echelon form (Demin, arXiv:2609.14433).
//!
//! Implements Algorithm 1 (Monagan-Pearce-Steel) over a prime field,
//! the success probability of Corollaries 1--3, and a Las-Vegas
//! certification that falls back to deterministic REF.

pub mod certify;
pub mod field;
pub mod matrix;
pub mod pref;

pub use certify::{all_rows_reduce_to_zero, certify_or_fallback, certified_ref, las_vegas_ref};
pub use field::Fp;
pub use matrix::{deterministic_ref, Matrix, RowEchelon};
pub use pref::{
    algorithm1, default_block_size, probabilistic_ref, proposition1_lower_bound,
    success_probability,
};
