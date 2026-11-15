//! Equation / invariant tests for arXiv:2609.19683 (MiX).

use mix_quant::{
    dr_ratio, factored_dot_type1, factored_dot_type2, implicit_mantissa, naive_dot,
    quantize_mantissa_3, reconstruct, sqnr_db, MixType1, MixType2, MxInt4, BLOCK, DELTA_ZERO,
    EBW_TYPE1, EBW_TYPE2, EMAX_BIAS, EMAX_MAX, EMAX_MIN, N_SUB, SUB_BLOCK, TYPE1_BLOCK_BITS,
    TYPE2_BLOCK_BITS,
};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn rand01(state: &mut u64) -> f32 {
    let u = splitmix64(state) >> 40; // 24 bits
    (u as f32) / ((1u32 << 24) as f32)
}

fn randn(state: &mut u64) -> f32 {
    // Box–Muller
    let u1 = rand01(state).max(1e-7);
    let u2 = rand01(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

fn close(a: f64, b: f64, tol: f64) {
    let scale = 1.0 + a.abs().max(b.abs());
    assert!(
        (a - b).abs() <= tol * scale,
        "expected {b}, got {a} (tol {tol})"
    );
}

fn fill_uniform(state: &mut u64, lo: f32, hi: f32) -> [f32; BLOCK] {
    let mut x = [0.0f32; BLOCK];
    for i in 0..BLOCK {
        x[i] = lo + (hi - lo) * rand01(state);
    }
    x
}

fn fill_gaussian(state: &mut u64, scale: f32) -> [f32; BLOCK] {
    let mut x = [0.0f32; BLOCK];
    for i in 0..BLOCK {
        x[i] = scale * randn(state);
    }
    x
}

/// Eq. (3)/(4) EBW constants.
#[test]
fn ebw_constants() {
    assert_eq!(TYPE1_BLOCK_BITS, 8 + 128);
    assert_eq!(TYPE1_BLOCK_BITS, 136);
    close(EBW_TYPE1, (TYPE1_BLOCK_BITS as f64) / (BLOCK as f64), 0.0);
    close(EBW_TYPE1, 4.25, 0.0);

    assert_eq!(TYPE2_BLOCK_BITS, 4 + 4 * 3 + 32 * 4);
    assert_eq!(TYPE2_BLOCK_BITS, 144);
    close(EBW_TYPE2, (TYPE2_BLOCK_BITS as f64) / (BLOCK as f64), 0.0);
    close(EBW_TYPE2, 4.5, 0.0);

    assert_eq!(N_SUB, 4);
    assert_eq!(SUB_BLOCK, 8);
    assert_eq!(EMAX_MAX - EMAX_MIN + 1, 16); // 4-bit field
    assert_eq!(EMAX_BIAS, 8);
}

#[test]
fn implicit_mantissa_eq3() {
    assert_eq!(implicit_mantissa(0), 1.0);
    assert_eq!(implicit_mantissa(4), 1.5);
    assert_eq!(implicit_mantissa(7), 1.875);
}

#[test]
fn delta_seven_is_zero() {
    assert_eq!(reconstruct(0, 7, 3, DELTA_ZERO), 0.0);
    assert_eq!(reconstruct(1, 3, -2, DELTA_ZERO), 0.0);
    let mut x = [1.0f32; BLOCK];
    x[3] = 0.0;
    x[9] = -0.0;
    x[17] = 0.0;
    let t1 = MixType1::quantize(&x);
    let t2 = MixType2::quantize(&x);
    assert_eq!(t1.delta_e[3], DELTA_ZERO);
    assert_eq!(t1.delta_e[9], DELTA_ZERO);
    assert_eq!(t1.dequantize()[3], 0.0);
    assert_eq!(t2.delta_e[17], DELTA_ZERO);
    assert_eq!(t2.dequantize()[17], 0.0);
}

#[test]
fn all_zero_block() {
    let x = [0.0f32; BLOCK];
    let t1 = MixType1::quantize(&x);
    let t2 = MixType2::quantize(&x);
    assert!(t1.delta_e.iter().all(|&d| d == DELTA_ZERO));
    assert!(t2.delta_e.iter().all(|&d| d == DELTA_ZERO));
    assert_eq!(t1.dequantize(), x);
    assert_eq!(t2.dequantize(), x);
}

#[test]
fn emax_bits_roundtrip() {
    let x = fill_gaussian(&mut 7, 1.5);
    let t1 = MixType1::quantize(&x);
    assert!(t1.emax >= EMAX_MIN && t1.emax <= EMAX_MAX);
    assert_eq!(t1.emax_bits() as i32 - EMAX_BIAS, t1.emax);
    assert!(t1.emax_bits() < 16);
}

/// `QuantizeMantissa` of the block max: 3-bit fraction, overflow increments Emax.
#[test]
fn quantize_mantissa_round_overflow() {
    // 1.0 → m=0, e=0
    assert_eq!(quantize_mantissa_3(1.0), (0, 0));
    // 1.5 = 1.100 → m=4, e=0
    assert_eq!(quantize_mantissa_3(1.5), (4, 0));
    // 1.875 = 1.1110 → m=7, e=0 (4th bit 0, no round)
    assert_eq!(quantize_mantissa_3(1.875), (7, 0));
    // 1.9375 = 1.1111 → 4th bit 1, overflow → m=0, e=1
    let (m, e) = quantize_mantissa_3(1.9375);
    assert_eq!((m, e), (0, 1));
}

fn assert_roundtrip(x: &[f32; BLOCK], name: &str) {
    for (label, xhat, delta, emax, m_at) in [
        {
            let q = MixType1::quantize(x);
            (
                "type1",
                q.dequantize(),
                q.delta_e,
                q.emax,
                [q.m_shared; BLOCK],
            )
        },
        {
            let q = MixType2::quantize(x);
            let mut m_at = [0u8; BLOCK];
            for i in 0..BLOCK {
                m_at[i] = q.m_shared[i / SUB_BLOCK];
            }
            ("type2", q.dequantize(), q.delta_e, q.emax, m_at)
        },
    ] {
        let _ = (emax, m_at);
        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..BLOCK {
            if x[i].abs() >= max_abs {
                max_abs = x[i].abs();
                max_idx = i;
            }
            if x[i] == 0.0 {
                assert_eq!(delta[i], DELTA_ZERO, "{name}/{label} zero Δe at {i}");
                assert_eq!(xhat[i], 0.0, "{name}/{label} zero reconstruct at {i}");
            }
            assert!(xhat[i].is_finite(), "{name}/{label} non-finite at {i}");
            // Peak-normalized absolute error is at most the peak itself
            // (Type 1 has only 7 magnitude rungs).
            if max_abs > 0.0 {
                assert!(
                    (xhat[i] - x[i]).abs() <= max_abs * 1.01 + 1e-6,
                    "{name}/{label} i={i}: x={} xhat={} max={}",
                    x[i],
                    xhat[i],
                    max_abs
                );
            }
            if delta[i] != DELTA_ZERO {
                assert!(delta[i] <= 6, "{name}/{label} nonzero Δe={}", delta[i]);
            }
        }
        if max_abs > 0.0 && x[max_idx] != 0.0 {
            let rel = (xhat[max_idx] - x[max_idx]).abs() / max_abs;
            assert!(
                rel < 0.20,
                "{name}/{label} max-element relative error {rel} (x={} xhat={})",
                x[max_idx],
                xhat[max_idx]
            );
        }
    }
}

#[test]
fn roundtrip_random_blocks() {
    let mut state = 0xC0FFEE_u64;
    assert_roundtrip(&[0.0; BLOCK], "zeros");

    let mut single = [0.0f32; BLOCK];
    single[11] = -3.25;
    assert_roundtrip(&single, "single");

    for n in 0..80 {
        let x = match n % 5 {
            0 => fill_uniform(&mut state, -1.0, 1.0),
            1 => fill_gaussian(&mut state, 2.0),
            2 => {
                let mut x = fill_uniform(&mut state, -4.0, 4.0);
                for i in 0..BLOCK {
                    if rand01(&mut state) < 0.15 {
                        x[i] = 0.0;
                    }
                }
                x
            }
            3 => {
                let mut x = fill_uniform(&mut state, 0.25, 1.5);
                x[0] = 40.0 + 20.0 * rand01(&mut state);
                x[7] = -x[0] * 0.5;
                x
            }
            _ => fill_uniform(&mut state, -0.05, 0.05),
        };
        assert_roundtrip(&x, &format!("block{n}"));
    }
}

/// Exponent finetuning never worse than the unrefined log-difference assignment.
#[test]
fn finetuning_never_worse() {
    let mut state = 0xABCDu64;
    for n in 0..60 {
        let x = if n % 3 == 0 {
            let mut x = fill_uniform(&mut state, 0.4, 1.6);
            x[n % BLOCK] = 12.0 + n as f32;
            x
        } else {
            fill_gaussian(&mut state, 1.8)
        };
        let t1_ft = MixType1::quantize(&x);
        let t1_nv = MixType1::quantize_unrefined(&x);
        let t2_ft = MixType2::quantize(&x);
        let t2_nv = MixType2::quantize_unrefined(&x);
        for i in 0..BLOCK {
            let xi = x[i] as f64;
            let e1_ft = (t1_ft.dequantize_at(i) - xi).abs();
            let e1_nv = (t1_nv.dequantize_at(i) - xi).abs();
            assert!(
                e1_ft <= e1_nv + 1e-15,
                "type1 i={i} ft={e1_ft} naive={e1_nv} x={}",
                x[i]
            );
            let e2_ft = (t2_ft.dequantize_at(i) - xi).abs();
            let e2_nv = (t2_nv.dequantize_at(i) - xi).abs();
            assert!(
                e2_ft <= e2_nv + 1e-15,
                "type2 i={i} ft={e2_ft} naive={e2_nv} x={}",
                x[i]
            );
        }
    }
}

fn gemm_pair(state: &mut u64, kind: u32) -> ([f32; BLOCK], [f32; BLOCK]) {
    let x = match kind % 4 {
        0 => fill_gaussian(state, 1.0),
        1 => {
            let mut x = fill_uniform(state, -2.0, 2.0);
            for i in 0..BLOCK {
                if rand01(state) < 0.2 {
                    x[i] = 0.0;
                }
            }
            x
        }
        2 => {
            let mut x = fill_uniform(state, 0.3, 1.1);
            x[0] = 50.0;
            x[15] = -25.0;
            x
        }
        _ => fill_uniform(state, -8.0, 8.0),
    };
    let w = fill_gaussian(state, 0.75);
    (x, w)
}

#[test]
fn factored_gemm_matches_naive() {
    let mut state = 0xF00D_u64;
    for n in 0..40 {
        let (x, w) = gemm_pair(&mut state, n);
        let t1 = MixType1::quantize(&x);
        let t2 = MixType2::quantize(&x);
        let mw = MxInt4::quantize(&w);
        let x1 = t1.dequantize_f64();
        let x2 = t2.dequantize_f64();
        let ww = mw.dequantize_f64();
        let naive1 = naive_dot(&x1, &ww);
        let naive2 = naive_dot(&x2, &ww);
        let fac1 = factored_dot_type1(&t1, &mw);
        let fac2 = factored_dot_type2(&t2, &mw);
        close(fac1, naive1, 1e-12);
        close(fac2, naive2, 1e-12);
    }

    // Hand-crafted zeros + mixed signs.
    let mut x = [0.0f32; BLOCK];
    let mut w = [0.0f32; BLOCK];
    x[0] = 1.0;
    x[1] = -2.0;
    x[2] = 0.0;
    x[3] = 0.5;
    w[0] = 3.0;
    w[1] = -1.0;
    w[2] = 9.0;
    w[3] = 0.25;
    let t1 = MixType1::quantize(&x);
    let mw = MxInt4::quantize(&w);
    close(
        factored_dot_type1(&t1, &mw),
        naive_dot(&t1.dequantize_f64(), &mw.dequantize_f64()),
        1e-12,
    );
}

/// Microscaling collapse: one huge outlier + small background. MiX keeps the
/// small values; MXINT4's shared exponent is pinned to the outlier.
#[test]
fn high_dynamic_range_mix_beats_mxint4_on_small_values() {
    let mut x = [0.0f32; BLOCK];
    for i in 0..BLOCK {
        x[i] = 0.25 + 0.05 * (i as f32); // ~0.25 … 1.80
    }
    x[0] = 80.0;

    let mix1 = MixType1::quantize(&x).dequantize();
    let mix2 = MixType2::quantize(&x).dequantize();
    let mx = MxInt4::quantize(&x).dequantize();

    let mse = |hat: &[f32; BLOCK]| -> f64 {
        let mut s = 0.0f64;
        for i in 1..BLOCK {
            let e = x[i] as f64 - hat[i] as f64;
            s += e * e;
        }
        s / (BLOCK - 1) as f64
    };
    let m1 = mse(&mix1);
    let m2 = mse(&mix2);
    let mm = mse(&mx);
    assert!(
        m1 < mm,
        "type1 small-value MSE {m1} should beat MXINT4 {mm}"
    );
    assert!(
        m2 < mm,
        "type2 small-value MSE {m2} should beat MXINT4 {mm}"
    );

    let mx_alive = (1..BLOCK).filter(|&i| mx[i] != 0.0).count();
    let mix_alive = (1..BLOCK).filter(|&i| mix1[i] != 0.0).count();
    assert!(
        mix_alive > mx_alive,
        "MiX should preserve more small values ({mix_alive}) than MXINT4 ({mx_alive})"
    );
    // MXINT4 shared scale is too coarse for the background.
    assert_eq!(mx_alive, 0, "MXINT4 should underflow the small values");
}

/// Build a block whose DR ratio `max/mean` is approximately `target`.
fn make_dr_block(target: f64, seed: u64) -> [f32; BLOCK] {
    let mut state = seed;
    let mut x = [0.0f32; BLOCK];
    for i in 1..BLOCK {
        x[i] = 0.7 + 0.6 * rand01(&mut state);
    }
    let mean_bg: f64 = x[1..].iter().map(|v| *v as f64).sum::<f64>() / (BLOCK - 1) as f64;
    // DR = 32 A / (31 μ + A)  ⇒  A = DR · 31 μ / (32 − DR)
    let a = target * (BLOCK - 1) as f64 * mean_bg / (BLOCK as f64 - target);
    x[0] = a as f32;
    x
}

/// Fig. 5 qualitative: MiX SQNR rises with DR ratio; MXINT4 collapses.
#[test]
fn sqnr_vs_dr_mix_rises_mxint4_collapses() {
    let targets = [2.5, 5.0, 8.0, 12.0, 16.0, 18.0];
    let mut mix_sqnr = Vec::new();
    let mut mx_sqnr = Vec::new();
    let mut drs = Vec::new();
    for (k, &t) in targets.iter().enumerate() {
        let x = make_dr_block(t, 0x1000 + k as u64);
        let dr = dr_ratio(&x);
        assert!(
            (dr - t).abs() < 0.35,
            "DR target {t} got {dr}"
        );
        let mix_hat = MixType2::quantize(&x).dequantize();
        let mx_hat = MxInt4::quantize(&x).dequantize();
        mix_sqnr.push(sqnr_db(&x, &mix_hat));
        mx_sqnr.push(sqnr_db(&x, &mx_hat));
        drs.push(dr);
    }

    // MiX: SQNR at high DR exceeds SQNR at low DR (steadily rising curve).
    assert!(
        mix_sqnr[4] > mix_sqnr[0] && mix_sqnr[5] > mix_sqnr[1],
        "MiX SQNR should increase with DR: dr={drs:?} sqnr={mix_sqnr:?}"
    );

    // MXINT4: collapse past the low-variance regime (paper: bottoms near 12 dB).
    assert!(
        mx_sqnr[2] < mx_sqnr[0] || mx_sqnr[3] < mx_sqnr[0],
        "MXINT4 SQNR should collapse at mid DR: {mx_sqnr:?}"
    );

    // At high DR, MiX is well above MXINT4.
    assert!(
        mix_sqnr[4] > mx_sqnr[4] && mix_sqnr[5] > mx_sqnr[5],
        "MiX should beat MXINT4 at high DR: mix={mix_sqnr:?} mx={mx_sqnr:?}"
    );
}

/// Eq. (3) reconstruction of a known block.
#[test]
fn eq3_hand_block() {
    let mut x = [0.0f32; BLOCK];
    x[0] = 4.0; // 1.0 · 2^2
    x[1] = -2.0;
    x[2] = 1.0;
    x[3] = 0.0;
    let q = MixType1::quantize(&x);
    assert_eq!(q.m_shared, 0); // mantissa of 4.0 is 1.000
    assert_eq!(q.emax, 2);
    assert_eq!(q.sign[1], 1);
    assert_eq!(q.delta_e[3], DELTA_ZERO);
    let hat = q.dequantize();
    close(hat[0] as f64, 4.0, 1e-6);
    close(hat[1] as f64, -2.0, 1e-6);
    close(hat[2] as f64, 1.0, 1e-6);
    close(hat[3] as f64, 0.0, 0.0);
}

/// Type 2 sub-blocks carry independent shared mantissas.
#[test]
fn type2_independent_sub_mantissas() {
    let mut x = [0.0f32; BLOCK];
    // sub 0: around 8 = 1.000 · 2^3
    for i in 0..8 {
        x[i] = 8.0;
    }
    // sub 1: around 6 = 1.100 · 2^2 → mantissa 4 if Emax from 8
    for i in 8..16 {
        x[i] = 6.0;
    }
    let q = MixType2::quantize(&x);
    assert_ne!(
        q.m_shared[0], q.m_shared[1],
        "distinct sub-block maxima should not share a mantissa ({:?})",
        q.m_shared
    );
}
