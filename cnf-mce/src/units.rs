//! Word-like units, span validity, and the MCE boundary penalty.
//!
//! A unit is a maximal run of Unicode letters or combining marks. CJK
//! ideographs are singleton units even though Unicode classifies them as
//! letters. Every other character is a singleton unit. Invalid UTF-8 bytes
//! are singleton units as well.

use std::cmp::Ordering;

include!("marks.rs");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanClass {
    /// Codepoint-aligned and allowed by the validity rule.
    Valid,
    /// Invalid, but a longer extension may become valid.
    Skip,
    /// Invalid, and no extension can become valid.
    Stop,
}

pub struct UnitIndex {
    unit_of: Vec<u32>,
    /// True at every codepoint start and at `bytes.len()`.
    boundary: Vec<bool>,
}

impl UnitIndex {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let n = bytes.len();
        let mut unit_of = vec![0u32; n];
        let mut boundary = vec![false; n + 1];
        boundary[0] = true;
        boundary[n] = true;
        if n == 0 {
            return Self { unit_of, boundary };
        }

        let mut i = 0;
        let mut unit_id: u32 = 0;
        let mut prev_run = false;
        let mut started = false;
        while i < n {
            boundary[i] = true;
            let (width, ch) = decode_one(&bytes[i..]);
            let run = ch.map(is_run_char).unwrap_or(false);
            let extend = started && run && prev_run;
            if !extend {
                if started {
                    unit_id = unit_id.saturating_add(1);
                }
                started = true;
            }
            for k in 0..width {
                unit_of[i + k] = unit_id;
            }
            i += width;
            prev_run = run;
        }
        boundary[n] = true;
        Self { unit_of, boundary }
    }

    pub fn is_char_boundary(&self, i: usize) -> bool {
        self.boundary.get(i).copied().unwrap_or(false)
    }

    pub fn is_unit_start(&self, i: usize) -> bool {
        i < self.unit_of.len() && (i == 0 || self.unit_of[i] != self.unit_of[i - 1])
    }

    pub fn is_unit_end(&self, i: usize) -> bool {
        i < self.unit_of.len()
            && (i + 1 == self.unit_of.len() || self.unit_of[i] != self.unit_of[i + 1])
    }

    pub fn touched(&self, start: usize, end: usize) -> u32 {
        if start >= end || end > self.unit_of.len() {
            return 0;
        }
        self.unit_of[end - 1] - self.unit_of[start] + 1
    }

    /// `None` means the span is excluded (+infinity).
    pub fn penalty(&self, start: usize, end: usize, lambda: f64) -> Option<f64> {
        let u = self.touched(start, end);
        if u == 0 {
            return None;
        }
        let aligned = self.is_unit_start(start) && self.is_unit_end(end - 1);
        if u == 1 || aligned {
            Some(0.0)
        } else if u == 2 {
            Some(lambda)
        } else {
            None
        }
    }

    pub fn classify(&self, start: usize, end: usize) -> SpanClass {
        if start >= end || end > self.unit_of.len() {
            return SpanClass::Stop;
        }
        if !self.is_char_boundary(start) {
            return SpanClass::Stop;
        }
        let starts_inside = !self.is_unit_start(start);
        let left_unit = starts_inside && self.unit_of[end - 1] != self.unit_of[start];
        if left_unit {
            return SpanClass::Stop;
        }
        if !self.is_char_boundary(end) {
            return SpanClass::Skip;
        }
        let u = self.touched(start, end);
        if u == 1 || self.is_unit_start(start) {
            SpanClass::Valid
        } else {
            SpanClass::Stop
        }
    }
}

pub fn is_valid_span(text: &str, start: usize, end: usize) -> bool {
    if start >= end || end > text.len() {
        return false;
    }
    let units = UnitIndex::from_bytes(text.as_bytes());
    units.classify(start, end) == SpanClass::Valid
}

pub fn boundary_penalty(text: &str, start: usize, end: usize, lambda: f64) -> Option<f64> {
    if start >= end || end > text.len() {
        return None;
    }
    let units = UnitIndex::from_bytes(text.as_bytes());
    units.penalty(start, end, lambda)
}

fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    COMBINING_MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                Ordering::Greater
            } else if cp > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

/// Han ideographs and a few alphabetic CJK marks. These are singleton units.
fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        0x3005..=0x3007
            | 0x3021..=0x3029
            | 0x3038..=0x303B
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2CEB0..=0x2EBEF
            | 0x2EBF0..=0x2EE5F
            | 0x2F800..=0x2FA1F
            | 0x30000..=0x3134F
            | 0x31350..=0x323AF
    )
}

fn is_run_char(c: char) -> bool {
    if is_combining_mark(c) {
        return true;
    }
    if is_cjk(c) {
        return false;
    }
    c.is_alphabetic()
}

/// Decode one Unicode scalar, or one invalid byte.
fn decode_one(bytes: &[u8]) -> (usize, Option<char>) {
    if bytes.is_empty() {
        return (1, None);
    }
    let b0 = bytes[0];
    if b0 < 0x80 {
        return (1, Some(b0 as char));
    }
    let width = match b0 {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return (1, None),
    };
    if bytes.len() < width {
        return (1, None);
    }
    match std::str::from_utf8(&bytes[..width]) {
        Ok(s) => (width, s.chars().next()),
        Err(_) => (1, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_matches_char_indices() {
        let samples = [
            "",
            "a",
            "é",
            "中文",
            "e\u{0301}x",
            "hello 世界",
            "co-advisor",
            "\u{1F600}!",
        ];
        for s in samples {
            let bytes = s.as_bytes();
            let units = UnitIndex::from_bytes(bytes);
            for i in 0..=bytes.len() {
                assert_eq!(
                    units.is_char_boundary(i),
                    s.is_char_boundary(i),
                    "boundary mismatch at {i} in {s:?}"
                );
            }
            let mut i = 0;
            for (idx, ch) in s.char_indices() {
                assert_eq!(i, idx);
                let (w, decoded) = decode_one(&bytes[i..]);
                assert_eq!(decoded, Some(ch));
                assert_eq!(w, ch.len_utf8());
                i += w;
            }
            assert_eq!(i, bytes.len());
        }
    }
}
