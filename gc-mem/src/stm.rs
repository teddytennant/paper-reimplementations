//! Optional short-term memory: FIFO of the last `N` chunks, unioned with MTM
//! top-k and deduplicated to form `R_k` (§4.1).

use std::collections::{HashSet, VecDeque};

use crate::types::{Chunk, Retrieved};

/// Recency buffer holding the `N` most recent ingested chunks.
#[derive(Clone, Debug)]
pub struct Stm<C> {
    capacity: usize,
    buf: VecDeque<Chunk<C>>,
}

impl<C> Stm<C> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buf: VecDeque::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn push(&mut self, chunk: Chunk<C>) {
        if self.capacity == 0 {
            return;
        }
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(chunk);
    }

    pub fn as_slice(&self) -> Vec<&Chunk<C>> {
        self.buf.iter().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Chunk<C>> {
        self.buf.iter()
    }
}

impl<C: Clone> Stm<C> {
    pub fn chunks(&self) -> Vec<Chunk<C>> {
        self.buf.iter().cloned().collect()
    }
}

/// Union MTM top-k with STM recent-N and drop duplicate ids, forming `R_k`.
///
/// MTM hits keep their retrieval relevance. STM-only chunks are appended with
/// `relevance = 1.0` (they were not scored by the bag ranker).
pub fn union_dedup<C: Clone>(mtm: &[Retrieved<C>], stm: &[Chunk<C>]) -> Vec<Retrieved<C>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for r in mtm {
        if seen.insert(r.chunk.id) {
            out.push(r.clone());
        }
    }
    for c in stm {
        if seen.insert(c.id) {
            out.push(Retrieved::new(c.clone(), 1.0));
        }
    }
    out
}
