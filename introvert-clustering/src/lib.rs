//! Introvert clustering and layered introvert network decomposition
//! (Chang and Dolatabadi, arXiv:2609.10044).
//!
//! Sequential randomized construction only: Theorem 7 (MPX), Lemma 1 (trim),
//! Lemma 2, and Theorem 3. No Ghaffari-Grunau deterministic construction.

mod decompose;
mod graph;
mod mpx;
mod trim;

pub use decompose::{introvert_cluster, layered_decompose, Layer, LayeredDecomp};
pub use graph::{validate_beta, validate_eps, validate_lambda, Clustering, Error, Graph};
pub use mpx::mpx_cluster;
pub use trim::{trim, trim_with_trace, RemovedVertex, TrimTrace};
