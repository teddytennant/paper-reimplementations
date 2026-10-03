//! Counting and Filtering vocabulary construction and Min-Cost Encoding.
//!
//! Counting is sequential. Summing document counts matches the paper's
//! parallel workers. Preference cost is section 4.1's formula, scaled by
//! eta (1 is the paper, 0 drops it).

mod units;
mod vocab;

pub use vocab::{preference_cost, Vocab};

/// Parameters for Counting and Filtering.
///
/// `length_intervals` are inclusive byte lengths, increasing, and cover 1.
/// `thresholds` has one entry per interval.
#[derive(Clone, Debug)]
pub struct CnfConfig {
    pub length_intervals: Vec<(usize, usize)>,
    pub thresholds: Vec<u64>,
    /// Raw vocabulary cap `m`, applied after all counting rounds.
    pub raw_vocab_cap: usize,
    /// Final vocabulary size `k`. If this is below 256, every single-byte
    /// token is still kept and the result records that.
    pub vocab_size: usize,
    /// Boundary penalty coefficient. The paper uses 0.05.
    pub lambda: f64,
    /// Scale on section 4.1's preference cost. `1.0` is the paper. `0.0` drops it.
    /// Callers pass this explicitly.
    pub eta: f64,
}

/// Build a vocabulary by counting valid spans, then filtering with MCE.
pub fn build_vocabulary(corpus: &[&str], cfg: &CnfConfig) -> Vocab {
    vocab::build_vocabulary(corpus, cfg)
}

/// Min-Cost Encoding of UTF-8 text.
/// `eta = 1` is section 4.1. `eta = 0` ignores token ids.
pub fn encode(text: &str, vocab: &Vocab, lambda: f64, eta: f64) -> Vec<u32> {
    vocab::encode(text, vocab, lambda, eta)
}

/// Min-Cost Encoding of an arbitrary byte string, including non-UTF-8 bytes.
pub fn encode_bytes(bytes: &[u8], vocab: &Vocab, lambda: f64, eta: f64) -> Vec<u32> {
    vocab::encode_bytes(bytes, vocab, lambda, eta)
}

/// Table lookup and concatenation. Encoding a `&str` and decoding yields that string.
pub fn decode(ids: &[u32], vocab: &Vocab) -> String {
    vocab::decode(ids, vocab)
}

/// Exact byte concatenation of the encoded tokens.
pub fn decode_bytes(ids: &[u32], vocab: &Vocab) -> Vec<u8> {
    vocab::decode_bytes(ids, vocab)
}

/// Byte indices, `end` exclusive.
///
/// Valid when both ends are codepoint boundaries and the span lies inside one
/// unit or starts at a unit boundary.
pub fn is_valid_span(text: &str, start: usize, end: usize) -> bool {
    units::is_valid_span(text, start, end)
}

/// Boundary penalty from the paper. `None` means +infinity (the span is excluded).
pub fn boundary_penalty(text: &str, start: usize, end: usize, lambda: f64) -> Option<f64> {
    units::boundary_penalty(text, start, end, lambda)
}
