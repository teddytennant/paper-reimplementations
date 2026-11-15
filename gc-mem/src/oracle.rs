//! Conflict function `κ(c_i, c_j) ∈ {0,1}` (§4.2).
//!
//! Pairwise NLI-style contradiction relative to a query. The paper's oracle is
//! an LLM NLI prompt; here it is an injected predicate.

use std::cell::Cell;

/// Pairwise contradiction oracle `κ`.
pub trait ConflictOracle<C> {
    /// `true` iff `a` and `b` logically contradict relative to `query`.
    fn contradicts(&self, a: &C, b: &C, query: &str) -> bool;
}

/// Exact structured-fact oracle for tests.
///
/// Facts contradict iff they share entity and attribute but differ in value.
/// The query is accepted to match the paper's `κ(·,·)` signature; structured
/// contradiction does not depend on surface wording.
#[derive(Clone, Debug, Default)]
pub struct ExactFactOracle;

impl ConflictOracle<crate::types::Fact> for ExactFactOracle {
    fn contradicts(&self, a: &crate::types::Fact, b: &crate::types::Fact, _query: &str) -> bool {
        a.entity == b.entity && a.attribute == b.attribute && a.value != b.value
    }
}

/// Wraps an oracle and counts `contradicts` calls (pairwise `O(|R_k|^2)`).
#[derive(Debug)]
pub struct CountingOracle<O> {
    inner: O,
    calls: Cell<u64>,
}

impl<O> CountingOracle<O> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            calls: Cell::new(0),
        }
    }

    pub fn calls(&self) -> u64 {
        self.calls.get()
    }

    pub fn reset(&self) {
        self.calls.set(0);
    }
}

impl<C, O: ConflictOracle<C>> ConflictOracle<C> for CountingOracle<O> {
    fn contradicts(&self, a: &C, b: &C, query: &str) -> bool {
        self.calls.set(self.calls.get() + 1);
        self.inner.contradicts(a, b, query)
    }
}
