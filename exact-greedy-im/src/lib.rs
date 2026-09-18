//! Exact Independent Cascade inference and exact greedy influence maximization
//! on graphs of treewidth at most 2, following arXiv:2609.19960.

pub mod autodiff;
pub mod graph;
pub mod greedy;
pub mod inference;
pub mod relations;
pub mod treewidth;

pub use graph::{Edge, Graph};
pub use greedy::exact_greedy;
pub use inference::{influence, influence_and_gradient};
pub use treewidth::TreeDecomposition;
