use frc_chunking::{pack, span_len, FrcParams};

fn covers(n: usize, chunks: &[frc_chunking::Chunk]) {
    let mut i = 0usize;
    for c in chunks {
        assert_eq!(c.start, i, "chunk starts at {i}, got {}", c.start);
        assert!(c.end > c.start, "empty chunk {c:?}");
        i = c.end;
    }
    assert_eq!(i, n, "chunks do not cover all {n} units");
}

fn params(m: f64, max: f64, delta: f64, k: usize) -> FrcParams {
    FrcParams {
        m,
        M: max,
        delta,
        k,
    }
}

#[test]
fn empty_input() {
    let r = pack(&[], &params(5.0, 10.0, 0.0, 1)).unwrap();
    assert!(r.chunks.is_empty());
    assert_eq!(r.cost, 0.0);
}

#[test]
fn each_unit_already_in_range_stays_separate_when_near_lstar() {
    // L* = 7.5. Units of length 7 are closer to L* than a pair of 14 (> M).
    let lens = [7usize, 7, 7];
    let r = pack(&lens, &params(5.0, 10.0, 0.0, 1)).unwrap();
    covers(3, &r.chunks);
    assert_eq!(r.stage, 1);
    assert_eq!(r.chunks.len(), 3);
    for c in &r.chunks {
        assert!(!c.violated);
        assert_eq!(c.end - c.start, 1);
    }
}

#[test]
fn packing_when_combining_is_cheaper() {
    // L* = 10. Two 6-length units: separate cost 2*(6-10)^2 = 32,
    // combined length 12, cost (12-10)^2 = 4.
    let lens = [6usize, 6];
    let r = pack(&lens, &params(5.0, 15.0, 0.0, 1)).unwrap();
    covers(2, &r.chunks);
    assert_eq!(r.stage, 1);
    assert_eq!(r.chunks.len(), 1);
    assert_eq!(r.chunks[0].start, 0);
    assert_eq!(r.chunks[0].end, 2);
    assert!((r.cost - 4.0).abs() < 1e-12);
}

#[test]
fn super_long_unit_is_isolated() {
    let lens = [3usize, 4, 100, 5];
    let r = pack(&lens, &params(5.0, 10.0, 0.0, 2)).unwrap();
    covers(4, &r.chunks);
    let super_long: Vec<_> = r
        .chunks
        .iter()
        .filter(|c| c.start == 2 && c.end == 3)
        .collect();
    assert_eq!(super_long.len(), 1);
    assert!(super_long[0].violated);
    assert!((super_long[0].length - 100.0).abs() < 1e-12);
    // Neighbours packed independently: 3+4=7 in range, then 5 in range.
    assert!(r.chunks.iter().any(|c| c.start == 0 && c.end == 2));
    assert!(r.chunks.iter().any(|c| c.start == 3 && c.end == 4));
}

#[test]
fn consecutive_super_long_units() {
    let lens = [80usize, 90];
    let r = pack(&lens, &params(5.0, 10.0, 0.0, 1)).unwrap();
    covers(2, &r.chunks);
    assert_eq!(r.chunks.len(), 2);
    assert!(r.chunks.iter().all(|c| c.violated && c.end - c.start == 1));
}

#[test]
fn stage1_infeasible_stage2_packs_with_violations() {
    // m=5, M=6, units [3,3,3], delta=0. No all-in-range partition.
    // Best with K>=1: [3,3]+[3] or [3]+[3,3], one violation, cost 0.25+4=4.25.
    let lens = [3usize, 3, 3];
    let r = pack(&lens, &params(5.0, 6.0, 0.0, 1)).unwrap();
    covers(3, &r.chunks);
    assert_eq!(r.stage, 2);
    assert!(r.cost.is_finite());
    assert!((r.cost - 4.25).abs() < 1e-12);
    assert_eq!(r.chunks.len(), 2);
    let vcount = r.chunks.iter().filter(|c| c.violated).count();
    assert_eq!(vcount, 1);
}

#[test]
fn stage2_never_infinite_when_k_at_least_one_and_a_partition_exists() {
    let lens = [1usize, 1, 1, 1];
    let r = pack(&lens, &params(10.0, 11.0, 0.0, 1)).unwrap();
    covers(4, &r.chunks);
    assert!(r.cost.is_finite());
    // Only option with 1 violation: the whole block as one short chunk.
    assert_eq!(r.chunks.len(), 1);
    assert!(r.chunks[0].violated);
}

#[test]
fn separator_cost_increases_multi_unit_length() {
    let lens = [3usize, 3];
    assert!((span_len(&lens, 0, 1, 0.0) - 6.0).abs() < 1e-12);
    assert!((span_len(&lens, 0, 1, 5.0) - 11.0).abs() < 1e-12);
    // delta=0: combined length 6 is in [5,10], so one chunk.
    let a = pack(&lens, &params(5.0, 10.0, 0.0, 1)).unwrap();
    assert_eq!(a.chunks.len(), 1);
    assert!(!a.chunks[0].violated);
    // delta=5: combined length 11 > 10, Stage 1 fails, Stage 2 emits the pair
    // as one violating chunk (cheaper than two length-3 violations).
    let b = pack(&lens, &params(5.0, 10.0, 5.0, 1)).unwrap();
    covers(2, &b.chunks);
    assert_eq!(b.stage, 2);
    assert!((span_len(&lens, 0, 1, 5.0) - 11.0).abs() < 1e-12);
}

#[test]
fn reconstruct_covers_every_unit_exactly_once_in_order() {
    let lens = [2usize, 3, 4, 5, 6, 7, 8];
    let r = pack(&lens, &params(5.0, 12.0, 1.0, 3)).unwrap();
    covers(lens.len(), &r.chunks);
    for (i, c) in r.chunks.iter().enumerate() {
        if i + 1 < r.chunks.len() {
            assert_eq!(c.end, r.chunks[i + 1].start);
        }
    }
}

/// Enumerate all 2^{n-1} partitions of `lens` (no super-long isolation).
fn brute_block_cost(lens: &[usize], p: &FrcParams) -> Option<f64> {
    let n = lens.len();
    if n == 0 {
        return Some(0.0);
    }
    let l_star = p.l_star();
    let limit = 1usize << (n - 1);
    let mut best_valid = f64::INFINITY;
    let mut best_rel = f64::INFINITY;
    for mask in 0..limit {
        let mut cost = 0.0;
        let mut viol = 0usize;
        let mut start = 0usize;
        for end in 0..n {
            let cut = end + 1 == n || (mask & (1 << end)) != 0;
            if !cut {
                continue;
            }
            let l = span_len(lens, start, end, p.delta);
            if l >= p.m && l <= p.M {
                cost += (l - l_star).powi(2);
            } else {
                viol += 1;
                cost += if l < p.m {
                    (p.m - l).powi(2)
                } else {
                    (l - p.M).powi(2)
                };
            }
            start = end + 1;
        }
        if viol == 0 {
            if cost < best_valid {
                best_valid = cost;
            }
        } else if viol <= p.k && cost < best_rel {
            best_rel = cost;
        }
    }
    if best_valid.is_finite() {
        Some(best_valid)
    } else if best_rel.is_finite() {
        Some(best_rel)
    } else {
        None
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn gen(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

#[test]
fn brute_force_oracle_n_le_8() {
    let mut rng = Lcg(0xD0C2_F1C4_u64);
    for _ in 0..80 {
        let n = rng.gen(1, 9);
        let mut lens = vec![0usize; n];
        for x in lens.iter_mut() {
            *x = rng.gen(1, 16);
        }
        let m = rng.gen(2, 8) as f64;
        let max = m + rng.gen(0, 8) as f64;
        let delta = rng.gen(0, 4) as f64;
        let k = rng.gen(1, 5);
        let p = params(m, max, delta, k);

        // Match pack() isolation: split on units longer than M, brute each block.
        let mut i = 0;
        let mut oracle = 0.0;
        let mut oracle_ok = true;
        while i < n {
            if (lens[i] as f64) > p.M {
                i += 1;
                continue;
            }
            let s = i;
            while i < n && (lens[i] as f64) <= p.M {
                i += 1;
            }
            match brute_block_cost(&lens[s..i], &p) {
                Some(c) => oracle += c,
                None => oracle_ok = false,
            }
        }

        match pack(&lens, &p) {
            Ok(r) => {
                covers(n, &r.chunks);
                assert!(oracle_ok, "DP found a pack but brute did not: {lens:?} {p:?}");
                assert!(
                    (r.cost - oracle).abs() < 1e-8,
                    "cost mismatch dp={} brute={} lens={lens:?} m={} M={} d={} k={}",
                    r.cost,
                    oracle,
                    p.m,
                    p.M,
                    p.delta,
                    p.k
                );
            }
            Err(_) => {
                assert!(!oracle_ok, "brute found a pack but DP failed: {lens:?} {p:?}");
            }
        }
    }
}

#[test]
fn single_short_unit_stage2() {
    let r = pack(&[3usize], &params(5.0, 10.0, 0.0, 1)).unwrap();
    covers(1, &r.chunks);
    assert_eq!(r.stage, 2);
    assert!(r.chunks[0].violated);
    assert!((r.cost - 4.0).abs() < 1e-12);
}

#[test]
fn k_zero_infeasible_when_stage1_fails() {
    let err = pack(&[3usize], &params(5.0, 10.0, 0.0, 0)).unwrap_err();
    match err {
        frc_chunking::PackError::Infeasible { block_units, .. } => {
            assert_eq!(block_units, 1);
        }
    }
}
