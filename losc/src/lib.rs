//! Logic Screening (LoSc) kernel.
//!
//! Faithful small-n implementation of the measurement model (Sec. II), pooling
//! designs (Sec. III-A), and decoding Algorithm 1 (Sec. III-B) from
//! <https://arxiv.org/abs/2609.21321>. Sample indices are 0-based.

mod decode;
mod error;
mod matrix;
mod measure;
mod pooling;

pub use decode::decode;
pub use error::LoscError;
pub use matrix::PoolingMatrix;
pub use measure::{dilution_matrix, measure};
pub use pooling::{sample_pooling, PoolingDesign};
