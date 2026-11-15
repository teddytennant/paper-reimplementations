//! Strong Connectivity Augmentation via a reduction to SCSS with two costs
//! (Koana and Kumabe, arXiv:2609.11160).
//!
//! Implements the Theorem 1.1 reduction to a terminal graph on at most 2k
//! vertices, Lemma 2.1 layered distances, the unweighted O*(4^k) ear DP
//! (Theorem 1.2), the weighted O*(9^k) ear DP (Theorem 1.3), and the
//! unweighted polynomial kernel of Theorem 1.4 without Frank-Tardos.

pub mod distance;
pub mod graph;
pub mod reduce;
pub mod sca;
pub mod scss;

pub use graph::DiGraph;
pub use reduce::{
    build_unweighted, build_weighted, kernelize_unweighted, TerminalGraph, WeightedTerminalGraph,
};
pub use sca::{sca_unweighted, sca_weighted, terminal_count};
pub use scss::{min_scss_cost, min_scss_weight, scss_unweighted, scss_weighted};
