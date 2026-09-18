//! GC-Mem: temporal dominance Φ_T over a retrieved RAG context (arXiv:2609.16073).
//!
//! Implements Definition 3.1, Assumption 3.3, Theorems 3.4–3.5, Equation (4),
//! and Theorem 4.1. The conflict oracle κ is an injected predicate; this crate
//! does not call an LLM.

pub mod baselines;
pub mod oracle;
pub mod phi;
pub mod stm;
pub mod theorems;
pub mod types;

pub use baselines::{
    multi_factor_score, recency, scored_dominance, timestamp_rerank, MultiFactorWeights,
};
pub use oracle::{ConflictOracle, CountingOracle, ExactFactOracle};
pub use phi::temporal_dominance;
pub use stm::{union_dedup, Stm};
pub use theorems::{
    count_vote, expected_valid_count, hypergeometric_draw, monte_carlo_mean_valid,
    proportional_attention, XorShift64,
};
pub use types::{Chunk, Fact, MemoryStream, Retrieved};
