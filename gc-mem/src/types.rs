//! Memory stream `M_t = {(c_i, τ_i)}` and retrieved items `m_i = (c_i, τ_i, r_i)`.

use std::cmp::Ordering;
use std::fmt;

/// Structured fact used by the exact κ oracle.
///
/// Two facts contradict iff they share entity and attribute but differ in value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fact {
    pub entity: String,
    pub attribute: String,
    pub value: String,
}

impl Fact {
    pub fn new(
        entity: impl Into<String>,
        attribute: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            entity: entity.into(),
            attribute: attribute.into(),
            value: value.into(),
        }
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}={}", self.entity, self.attribute, self.value)
    }
}

/// Memory chunk `(c_i, τ_i)` from the append-only stream `M_t` (§3).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Chunk<C> {
    pub id: u64,
    pub content: C,
    pub timestamp: u64,
}

impl<C> Chunk<C> {
    pub fn new(id: u64, content: C, timestamp: u64) -> Self {
        Self {
            id,
            content,
            timestamp,
        }
    }
}

/// Retrieved item `m_i = (c_i, τ_i, r_i)` (§4.2).
///
/// `usage` and `stability` are extra fields for the multi-factor ablation (§6.4);
/// they are not part of the GC-Mem tuple.
#[derive(Clone, Debug)]
pub struct Retrieved<C> {
    pub chunk: Chunk<C>,
    pub relevance: f64,
    pub usage: f64,
    pub stability: f64,
}

impl<C> Retrieved<C> {
    pub fn new(chunk: Chunk<C>, relevance: f64) -> Self {
        Self {
            chunk,
            relevance,
            usage: 0.0,
            stability: 0.0,
        }
    }

    pub fn with_usage_stability(mut self, usage: f64, stability: f64) -> Self {
        self.usage = usage;
        self.stability = stability;
        self
    }
}

impl<C: PartialEq> PartialEq for Retrieved<C> {
    fn eq(&self, other: &Self) -> bool {
        self.chunk == other.chunk
            && self.relevance == other.relevance
            && self.usage == other.usage
            && self.stability == other.stability
    }
}

/// Append-only in-memory bag. Not a dense retriever.
#[derive(Clone, Debug, Default)]
pub struct MemoryStream<C> {
    items: Vec<Chunk<C>>,
}

impl<C> MemoryStream<C> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn append(&mut self, chunk: Chunk<C>) {
        self.items.push(chunk);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn as_slice(&self) -> &[Chunk<C>] {
        &self.items
    }
}

impl<C: Clone> MemoryStream<C> {
    /// Rank by `relevance` descending and keep top-`k` (bag, not embeddings).
    pub fn top_k(&self, k: usize, mut relevance: impl FnMut(&C) -> f64) -> Vec<Retrieved<C>> {
        let mut v: Vec<Retrieved<C>> = self
            .items
            .iter()
            .map(|c| Retrieved::new(c.clone(), relevance(&c.content)))
            .collect();
        v.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(Ordering::Equal)
                .then(b.chunk.timestamp.cmp(&a.chunk.timestamp))
                .then(a.chunk.id.cmp(&b.chunk.id))
        });
        v.truncate(k);
        v
    }
}
