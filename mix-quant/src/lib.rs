//! Micro-Inverted-Scaling (MiX) format kernel
//! (Liao & Seo, [arXiv:2609.19683](https://arxiv.org/abs/2609.19683)).
//!
//! A block shares one mantissa; each element has a private relative exponent
//! (paper §III). Software format only — no CMOS / systolic accelerator.

pub const BLOCK: usize = 32;
pub const SUB_BLOCK: usize = 8;
pub const N_SUB: usize = BLOCK / SUB_BLOCK; // K = 4

/// 3-bit shared mantissa with implicit leading 1.
pub const MANT_BITS: u32 = 3;
/// 4-bit block-maximum exponent.
pub const EMAX_BITS: u32 = 4;

/// Stored `emax_bits = Emax + EMAX_BIAS`, so `Emax ∈ [EMAX_MIN, EMAX_MAX]`.
///
/// The paper (Alg. 1) extracts `Emax = ⌊log2(max |x|)⌋` but does not specify
/// the 4-bit encoding. Bias 8 covers typical VLM activations quoted in §I
/// (vision ≈ [−20, 10], text near 0) plus a few extra integer powers of two.
pub const EMAX_BIAS: i32 = 8;
pub const EMAX_MIN: i32 = -8;
pub const EMAX_MAX: i32 = 7;

/// `Δe_i = 7` encodes zero (§III-A). Nonzero elements use `Δe ∈ {0,…,6}`.
pub const DELTA_ZERO: u8 = 7;

/// Type 1: 4+3 shared bits padded with one spare to an 8-bit field, plus
/// 128 element bits. `(8 + 128) / 32 = 4.25` (§III-A).
pub const TYPE1_SHARED_BITS: u32 = 8;
pub const TYPE1_ELEM_BITS: u32 = 4;
pub const TYPE1_BLOCK_BITS: u32 = TYPE1_SHARED_BITS + (BLOCK as u32) * TYPE1_ELEM_BITS;
pub const EBW_TYPE1: f64 = 4.25;

/// Type 2: `4 + 4×3 + 32×4 = 144` bits, EBW 4.5 (§III-A).
pub const TYPE2_BLOCK_BITS: u32 = 4 + (N_SUB as u32) * 3 + (BLOCK as u32) * 4;
pub const EBW_TYPE2: f64 = 4.5;

/// `(1.m_shared)_2 = 1 + m_shared/8` (paper §III-A).
#[inline]
pub fn implicit_mantissa(m_shared: u8) -> f64 {
    debug_assert!(m_shared < 8);
    1.0 + (m_shared as f64) / 8.0
}

/// SQNR = `10 log10( Σ x² / Σ (x − x̂)² )` (paper §III-C).
pub fn sqnr_db(x: &[f32], xhat: &[f32]) -> f64 {
    assert_eq!(x.len(), xhat.len());
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&a, &b) in x.iter().zip(xhat) {
        let af = a as f64;
        let e = af - b as f64;
        sig += af * af;
        noise += e * e;
    }
    if noise == 0.0 {
        f64::INFINITY
    } else if sig == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (sig / noise).log10()
    }
}

/// Intra-block dynamic-range ratio `max |x| / mean |x|` (Fig. 5, x-axis).
pub fn dr_ratio(x: &[f32]) -> f64 {
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for &v in x {
        let a = (v as f64).abs();
        if a > max {
            max = a;
        }
        sum += a;
    }
    let mean = sum / x.len() as f64;
    if mean == 0.0 {
        0.0
    } else {
        max / mean
    }
}

/// Reconstruct one element (eqs. 3–4). `delta == 7` is zero.
#[inline]
pub fn reconstruct(sign: u8, m_shared: u8, emax: i32, delta: u8) -> f64 {
    if delta == DELTA_ZERO {
        return 0.0;
    }
    let mag = implicit_mantissa(m_shared) * 2f64.powi(emax - delta as i32);
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

/// Round `abs_x`’s mantissa to 3 explicit bits (Alg. 1 `QuantizeMantissa`).
///
/// Hardware (§IV-C): take 4 MSBs of the fraction, add the 4th bit into the
/// upper 3; on overflow (`111₂ + 1 = 1000₂`) increment the exponent and
/// leave `m_shared = 000₂`.
pub fn quantize_mantissa_3(abs_x: f32) -> (u8, i32) {
    debug_assert!(abs_x > 0.0 && abs_x.is_finite());
    let bits = abs_x.to_bits();
    let efield = (bits >> 23) & 0xff;
    if efield == 0 || efield == 0xff {
        return (0, EMAX_MIN);
    }
    let mut exp = efield as i32 - 127;
    let frac = bits & 0x7f_ffff;
    let mut m = (frac >> 20) as u8; // top 3 fraction bits
    let round_bit = (frac >> 19) & 1; // 4th fraction bit
    if round_bit == 1 {
        m += 1;
        if m == 8 {
            m = 0;
            exp += 1;
        }
    }
    (m, exp)
}

fn floor_log2(x: f32) -> i32 {
    debug_assert!(x > 0.0 && x.is_finite());
    let efield = (x.to_bits() >> 23) & 0xff;
    if efield == 0 {
        // Subnormals sit at ≤ 2^{−126}, far below [`EMAX_MIN`].
        return -149;
    }
    efield as i32 - 127
}

fn clamp_emax(e: i32) -> i32 {
    e.clamp(EMAX_MIN, EMAX_MAX)
}

/// Alg. 1 lines 7–11: sign, naive log-difference, optional ±1 finetune.
fn assign_delta(xi: f32, emax: i32, m_shared: u8, finetune: bool) -> (u8, u8) {
    if !xi.is_finite() || xi == 0.0 {
        return (0, DELTA_ZERO);
    }
    let sign = if xi < 0.0 { 1 } else { 0 };
    let ax = xi.abs();
    let naive_raw = emax - floor_log2(ax);
    // Values ≥ 2^{7} smaller than 2^{Emax} underflow to the zero code.
    if naive_raw >= 7 {
        return (sign, DELTA_ZERO);
    }
    let naive = naive_raw.max(0).min(6);
    if !finetune {
        return (sign, naive as u8);
    }
    // Search {Δe−1, Δe, Δe+1} ∩ {0,…,6} (Alg. 1 lines 9–11, clamped so 7
    // stays reserved for zero).
    let lo = (naive - 1).max(0);
    let hi = (naive + 1).min(6);
    let x64 = xi as f64;
    let mut best = naive;
    let mut best_err = f64::INFINITY;
    for d in lo..=hi {
        let err = (reconstruct(sign, m_shared, emax, d as u8) - x64).abs();
        if err < best_err {
            best_err = err;
            best = d;
        }
    }
    (sign, best as u8)
}

fn block_max_abs(x: &[f32; BLOCK]) -> f32 {
    let mut m = 0.0f32;
    for &v in x {
        if v.is_finite() {
            m = m.max(v.abs());
        }
    }
    m
}

/// Type 1 MiX-4.25b (§III-A, eq. 3): one 4-bit `Emax`, one 3-bit `m_shared`,
/// per element 1-bit sign + 3-bit `Δe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixType1 {
    /// Unbiased block-maximum exponent, in `[EMAX_MIN, EMAX_MAX]`.
    pub emax: i32,
    /// 3-bit shared mantissa `0..7`.
    pub m_shared: u8,
    pub sign: [u8; BLOCK],
    pub delta_e: [u8; BLOCK],
}

impl MixType1 {
    /// Packed 4-bit Emax field (`Emax + EMAX_BIAS`).
    pub fn emax_bits(self) -> u8 {
        (self.emax + EMAX_BIAS) as u8
    }

    /// Algorithm 1 with exponent finetuning (one shared mantissa for B=32).
    pub fn quantize(x: &[f32; BLOCK]) -> Self {
        Self::quantize_with(x, true)
    }

    /// Same as [`quantize`](Self::quantize) but skip the ±1 search (line 8 only).
    pub fn quantize_unrefined(x: &[f32; BLOCK]) -> Self {
        Self::quantize_with(x, false)
    }

    pub fn quantize_with(x: &[f32; BLOCK], finetune: bool) -> Self {
        let max_abs = block_max_abs(x);
        if max_abs == 0.0 || max_abs < 2f32.powi(EMAX_MIN) {
            return Self {
                emax: EMAX_MIN,
                m_shared: 0,
                sign: [0; BLOCK],
                delta_e: [DELTA_ZERO; BLOCK],
            };
        }
        let (m_shared, emax) = quantize_mantissa_3(max_abs);
        let emax = clamp_emax(emax);
        let mut sign = [0u8; BLOCK];
        let mut delta_e = [DELTA_ZERO; BLOCK];
        for i in 0..BLOCK {
            let (s, d) = assign_delta(x[i], emax, m_shared, finetune);
            sign[i] = s;
            delta_e[i] = d;
        }
        Self {
            emax,
            m_shared,
            sign,
            delta_e,
        }
    }

    pub fn dequantize_at(self, i: usize) -> f64 {
        reconstruct(self.sign[i], self.m_shared, self.emax, self.delta_e[i])
    }

    pub fn dequantize_f64(self) -> [f64; BLOCK] {
        let mut y = [0.0f64; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i);
        }
        y
    }

    /// Eq. (3).
    pub fn dequantize(self) -> [f32; BLOCK] {
        let mut y = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i) as f32;
        }
        y
    }
}

/// Type 2 MiX-4.5b (§III-A, eq. 4): global 4-bit `Emax`, K=4 sub-blocks of
/// b=8 each with its own 3-bit `m_shared,k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixType2 {
    pub emax: i32,
    pub m_shared: [u8; N_SUB],
    pub sign: [u8; BLOCK],
    pub delta_e: [u8; BLOCK],
}

impl MixType2 {
    pub fn emax_bits(self) -> u8 {
        (self.emax + EMAX_BIAS) as u8
    }

    pub fn quantize(x: &[f32; BLOCK]) -> Self {
        Self::quantize_with(x, true)
    }

    pub fn quantize_unrefined(x: &[f32; BLOCK]) -> Self {
        Self::quantize_with(x, false)
    }

    pub fn quantize_with(x: &[f32; BLOCK], finetune: bool) -> Self {
        let max_abs = block_max_abs(x);
        if max_abs == 0.0 || max_abs < 2f32.powi(EMAX_MIN) {
            return Self {
                emax: EMAX_MIN,
                m_shared: [0; N_SUB],
                sign: [0; BLOCK],
                delta_e: [DELTA_ZERO; BLOCK],
            };
        }
        // Global Emax from the block maximum (Alg. 1 line 2), including the
        // mantissa-rounding overflow increment of §IV-C.
        let (_m_global, emax) = quantize_mantissa_3(max_abs);
        let emax = clamp_emax(emax);
        let mut m_shared = [0u8; N_SUB];
        let mut sign = [0u8; BLOCK];
        let mut delta_e = [DELTA_ZERO; BLOCK];
        for k in 0..N_SUB {
            let lo = k * SUB_BLOCK;
            let hi = lo + SUB_BLOCK;
            let mut sub_max = 0.0f32;
            for v in &x[lo..hi] {
                if v.is_finite() {
                    sub_max = sub_max.max(v.abs());
                }
            }
            let mk = if sub_max > 0.0 {
                quantize_mantissa_3(sub_max).0
            } else {
                0
            };
            m_shared[k] = mk;
            for i in lo..hi {
                let (s, d) = assign_delta(x[i], emax, mk, finetune);
                sign[i] = s;
                delta_e[i] = d;
            }
        }
        Self {
            emax,
            m_shared,
            sign,
            delta_e,
        }
    }

    pub fn dequantize_at(self, i: usize) -> f64 {
        let k = i / SUB_BLOCK;
        reconstruct(self.sign[i], self.m_shared[k], self.emax, self.delta_e[i])
    }

    pub fn dequantize_f64(self) -> [f64; BLOCK] {
        let mut y = [0.0f64; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i);
        }
        y
    }

    /// Eq. (4).
    pub fn dequantize(self) -> [f32; BLOCK] {
        let mut y = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i) as f32;
        }
        y
    }
}

/// MXINT4-style baseline (Fig. 4b): 8-bit shared exponent, per-element
/// 1-bit sign + 3-bit integer mantissa, block 32.
///
/// `ŵ_i = (−1)^{s_i} · m_i · 2^{e_shared}`, `m_i ∈ {0,…,7}`.
/// `e_shared` is chosen so the block maximum maps onto the 3-bit range
/// (`7 · 2^{e_shared} ≥ max |w|`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MxInt4 {
    pub e_shared: i32,
    pub sign: [u8; BLOCK],
    pub mantissa: [u8; BLOCK],
}

impl MxInt4 {
    pub fn quantize(w: &[f32; BLOCK]) -> Self {
        let max_abs = block_max_abs(w);
        if max_abs == 0.0 {
            return Self {
                e_shared: 0,
                sign: [0; BLOCK],
                mantissa: [0; BLOCK],
            };
        }
        // Smallest integer E with 7 · 2^E ≥ max|w|.
        let e_shared = (max_abs as f64 / 7.0).log2().ceil() as i32;
        let scale = 2f64.powi(-e_shared);
        let mut sign = [0u8; BLOCK];
        let mut mantissa = [0u8; BLOCK];
        for i in 0..BLOCK {
            let v = w[i];
            if !v.is_finite() || v == 0.0 {
                continue;
            }
            sign[i] = if v < 0.0 { 1 } else { 0 };
            let m = ((v.abs() as f64) * scale).round() as i32;
            mantissa[i] = m.clamp(0, 7) as u8;
        }
        Self {
            e_shared,
            sign,
            mantissa,
        }
    }

    pub fn dequantize_at(self, i: usize) -> f64 {
        let mag = (self.mantissa[i] as f64) * 2f64.powi(self.e_shared);
        if self.sign[i] == 1 {
            -mag
        } else {
            mag
        }
    }

    pub fn dequantize_f64(self) -> [f64; BLOCK] {
        let mut y = [0.0f64; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i);
        }
        y
    }

    pub fn dequantize(self) -> [f32; BLOCK] {
        let mut y = [0.0f32; BLOCK];
        for i in 0..BLOCK {
            y[i] = self.dequantize_at(i) as f32;
        }
        y
    }
}

/// Naive dequant-then-dot in f64.
pub fn naive_dot(xhat: &[f64; BLOCK], what: &[f64; BLOCK]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..BLOCK {
        acc += xhat[i] * what[i];
    }
    acc
}

/// Factored MiX-MX block GEMM for Type 1 activations (eq. 6):
///
/// `y = (1.m_shared)_2 · 2^{Emax + E_w} · Σ_i [(−1)^{s_x,i ⊕ s_w,i} · m_w,i · 2^{−Δe_x,i}]`
///
/// Element-wise work is a right-shift of the weight mantissa by `Δe_x,i`,
/// then one scale of the accumulated sum. `Δe = 7` contributes zero.
pub fn factored_dot_type1(x: &MixType1, w: &MxInt4) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..BLOCK {
        if x.delta_e[i] == DELTA_ZERO || w.mantissa[i] == 0 {
            continue;
        }
        let mut mw = w.mantissa[i] as f64;
        if (x.sign[i] ^ w.sign[i]) == 1 {
            mw = -mw;
        }
        acc += mw * 2f64.powi(-(x.delta_e[i] as i32));
    }
    implicit_mantissa(x.m_shared) * 2f64.powi(x.emax + w.e_shared) * acc
}

/// Type 2 analogue of eq. 6: one scale per sub-block mantissa, shared
/// `2^{Emax + E_w}` across the block.
pub fn factored_dot_type2(x: &MixType2, w: &MxInt4) -> f64 {
    let pow = 2f64.powi(x.emax + w.e_shared);
    let mut y = 0.0f64;
    for k in 0..N_SUB {
        let mut acc = 0.0f64;
        let lo = k * SUB_BLOCK;
        let hi = lo + SUB_BLOCK;
        for i in lo..hi {
            if x.delta_e[i] == DELTA_ZERO || w.mantissa[i] == 0 {
                continue;
            }
            let mut mw = w.mantissa[i] as f64;
            if (x.sign[i] ^ w.sign[i]) == 1 {
                mw = -mw;
            }
            acc += mw * 2f64.powi(-(x.delta_e[i] as i32));
        }
        y += implicit_mantissa(x.m_shared[k]) * acc;
    }
    y * pow
}
