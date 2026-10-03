//! Vocabulary, Counting and Filtering, and Min-Cost Encoding.

use std::collections::HashMap;

use crate::units::{SpanClass, UnitIndex};
use crate::CnfConfig;

/// Token strings, ids, and a byte-string lookup.
///
/// Id 0 is the most-used token after filtering. Unused single-byte tokens
/// receive the largest ids, ordered by byte value.
#[derive(Clone, Debug)]
pub struct Vocab {
    tokens: Vec<Vec<u8>>,
    index: HashMap<Vec<u8>, u32>,
    max_len: usize,
    /// Set when `CnfConfig::vocab_size` was below 256.
    /// All 256 single-byte tokens are still kept.
    pub recorded_vocab_size_below_256: bool,
    /// Vocabulary size requested by the caller. Zero for hand-built vocabs.
    pub requested_vocab_size: usize,
}

impl Vocab {
    /// Ids follow `tokens` order. Missing single-byte tokens are appended in
    /// byte order so every byte string has a segmentation.
    pub fn from_tokens<I, T>(tokens: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let mut out = Vec::new();
        let mut seen = HashMap::new();
        for token in tokens {
            let token = token.as_ref();
            if token.is_empty() || seen.contains_key(token) {
                continue;
            }
            let id = out.len() as u32;
            seen.insert(token.to_vec(), id);
            out.push(token.to_vec());
        }
        for b in 0u16..=255 {
            let token = vec![b as u8];
            if seen.contains_key(&token) {
                continue;
            }
            let id = out.len() as u32;
            seen.insert(token.clone(), id);
            out.push(token);
        }
        Self::from_ordered(out, false, 0)
    }

    fn from_ordered(tokens: Vec<Vec<u8>>, small_cap: bool, requested: usize) -> Self {
        let mut index = HashMap::with_capacity(tokens.len());
        let mut max_len = 1usize;
        for (id, token) in tokens.iter().enumerate() {
            max_len = max_len.max(token.len());
            index.insert(token.clone(), id as u32);
        }
        Self {
            tokens,
            index,
            max_len,
            recorded_vocab_size_below_256: small_cap,
            requested_vocab_size: requested,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn get_id(&self, token: &[u8]) -> Option<u32> {
        self.index.get(token).copied()
    }

    pub fn token(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(Vec::as_slice)
    }

    pub fn contains(&self, token: &[u8]) -> bool {
        self.index.contains_key(token)
    }

    pub fn max_token_len(&self) -> usize {
        self.max_len
    }
}

pub fn build_vocabulary(corpus: &[&str], cfg: &CnfConfig) -> Vocab {
    let intervals = &cfg.length_intervals;
    let mut retained: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut processed: Vec<(usize, usize)> = Vec::new();

    // Sequential pass over documents. Summing per-document counts matches the
    // paper's parallel workers.
    for (round, &(rmin, rmax)) in intervals.iter().enumerate() {
        if rmax == 0 || rmin > rmax {
            continue;
        }
        let rmin = rmin.max(1);
        let threshold = cfg.thresholds.get(round).copied().unwrap_or(0);
        let mut round_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for doc in corpus {
            let local = count_document(doc, rmin, rmax, &processed, &retained, threshold);
            for (token, count) in local {
                *round_counts.entry(token).or_default() += count;
            }
        }
        for (token, count) in round_counts {
            if count >= threshold {
                retained.entry(token).or_insert(count);
            }
        }
        processed.push((rmin, rmax));
    }

    let mut ranked: Vec<(Vec<u8>, u64)> = retained.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.0.len().cmp(&a.0.len()))
            .then(a.0.cmp(&b.0))
    });
    if ranked.len() > cfg.raw_vocab_cap {
        ranked.truncate(cfg.raw_vocab_cap);
    }

    // Raw vocabulary: top-m candidates, plus every single-byte token.
    let raw_pieces: Vec<Vec<u8>> = ranked.into_iter().map(|(token, _)| token).collect();
    let raw = Vocab::from_tokens(raw_pieces);

    let mut seg_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    for doc in corpus {
        let ids = encode(doc, &raw, cfg.lambda, cfg.eta);
        for id in ids {
            if let Some(token) = raw.token(id) {
                *seg_counts.entry(token.to_vec()).or_default() += 1;
            }
        }
    }

    let small_cap = cfg.vocab_size < 256;
    let slots = if small_cap { 0 } else { cfg.vocab_size - 256 };
    let mut non_byte: Vec<(Vec<u8>, u64)> = raw
        .tokens
        .iter()
        .filter(|token| token.len() > 1)
        .filter_map(|token| {
            let count = seg_counts.get(token).copied().unwrap_or(0);
            if count > 0 {
                Some((token.clone(), count))
            } else {
                None
            }
        })
        .collect();
    non_byte.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if non_byte.len() > slots {
        non_byte.truncate(slots);
    }

    let mut used: Vec<(Vec<u8>, u64)> = Vec::new();
    let mut unused_bytes: Vec<u8> = Vec::new();
    for b in 0u16..=255 {
        let token = vec![b as u8];
        let count = seg_counts.get(&token).copied().unwrap_or(0);
        if count > 0 {
            used.push((token, count));
        } else {
            unused_bytes.push(b as u8);
        }
    }
    used.extend(non_byte);
    used.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    unused_bytes.sort_unstable();

    let mut ordered = Vec::with_capacity(used.len() + unused_bytes.len());
    for (token, _) in used {
        ordered.push(token);
    }
    for b in unused_bytes {
        ordered.push(vec![b]);
    }
    Vocab::from_ordered(ordered, small_cap, cfg.vocab_size)
}

fn count_document(
    doc: &str,
    rmin: usize,
    rmax: usize,
    processed: &[(usize, usize)],
    retained: &HashMap<Vec<u8>, u64>,
    threshold: u64,
) -> HashMap<Vec<u8>, u64> {
    let bytes = doc.as_bytes();
    let n = bytes.len();
    let units = UnitIndex::from_bytes(bytes);
    let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
    if n == 0 || rmax == 0 {
        return counts;
    }
    for i in 0..n {
        if !units.is_char_boundary(i) {
            continue;
        }
        let limit = rmax.min(n - i);
        for len in 1..=limit {
            let end = i + len;
            match units.classify(i, end) {
                SpanClass::Stop => break,
                SpanClass::Skip => continue,
                SpanClass::Valid => {
                    if interval_contains(processed, len) {
                        // Algorithm 1: missing from the retained trie, or stored
                        // frequency below the current threshold.
                        match retained.get(&bytes[i..end]) {
                            Some(&count) if count >= threshold => {}
                            _ => break,
                        }
                    } else if len >= rmin && len <= rmax {
                        *counts.entry(bytes[i..end].to_vec()).or_default() += 1;
                    }
                }
            }
        }
    }
    counts
}

fn interval_contains(intervals: &[(usize, usize)], len: usize) -> bool {
    intervals.iter().any(|&(lo, hi)| len >= lo && len <= hi)
}

pub fn encode(text: &str, vocab: &Vocab, lambda: f64, eta: f64) -> Vec<u32> {
    encode_bytes(text.as_bytes(), vocab, lambda, eta)
}

pub fn encode_bytes(bytes: &[u8], vocab: &Vocab, lambda: f64, eta: f64) -> Vec<u32> {
    let n = bytes.len();
    if n == 0 {
        return Vec::new();
    }
    let units = UnitIndex::from_bytes(bytes);
    let vocab_len = vocab.len().max(1) as f64;
    let mut dp = vec![0.0; n + 1];
    let mut next = vec![0usize; n];
    for i in (0..n).rev() {
        let limit = vocab.max_len.min(n - i);
        let mut best_cost = f64::INFINITY;
        let mut best_j = i + 1;
        // Descending j so an equal cost keeps the largest j.
        for len in (1..=limit).rev() {
            let j = i + len;
            let Some(&id) = vocab.index.get(&bytes[i..j]) else {
                continue;
            };
            let Some(penalty) = units.penalty(i, j, lambda) else {
                continue;
            };
            let eps = eta * (id as f64) / vocab_len;
            let total = 1.0 + penalty + eps + dp[j];
            if total < best_cost {
                best_cost = total;
                best_j = j;
            }
        }
        if !best_cost.is_finite() {
            // Bytes are always in V, so this is only a fallback.
            best_j = i + 1;
            best_cost = 1.0 + dp[best_j];
        }
        dp[i] = best_cost;
        next[i] = best_j;
    }

    let mut ids = Vec::new();
    let mut i = 0;
    while i < n {
        let j = next[i];
        if j > i {
            if let Some(&id) = vocab.index.get(&bytes[i..j]) {
                ids.push(id);
                i = j;
                continue;
            }
        }
        // Single-byte fallback. Every byte is in V, so this still round-trips.
        let id = vocab.index.get(&bytes[i..i + 1]).copied().unwrap_or(0);
        ids.push(id);
        i += 1;
    }
    ids
}

pub fn decode_bytes(ids: &[u32], vocab: &Vocab) -> Vec<u8> {
    let mut out = Vec::new();
    for &id in ids {
        if let Some(token) = vocab.token(id) {
            out.extend_from_slice(token);
        }
    }
    out
}

pub fn decode(ids: &[u32], vocab: &Vocab) -> String {
    let bytes = decode_bytes(ids, vocab);
    match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(err) => String::from_utf8_lossy(err.as_bytes()).into_owned(),
    }
}
