//! A seeded splitmix64 generator.
//!
//! The crate has no dependencies, so the randomness used by DCluster is
//! hand-rolled here. splitmix64 is the standard finalizer from Steele, Lea and
//! Flood's SplittableRandom; it has a 64-bit state, a period of 2^64, and it is
//! good enough for the independent coin flips the algorithm asks for. Every
//! stream is seeded explicitly, so two runs with the same seed produce the same
//! clustering, the same message count and the same round count.

/// A splitmix64 stream.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Start a stream at `seed`.
    pub fn seed(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, built from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// One coin flip that comes up true with probability `p`.
    ///
    /// `p >= 1.0` and `p <= 0.0` short-circuit without drawing, which matters
    /// for reproducibility: DCluster's sampling probability is
    /// `min{1, 6 ln n / Delta_i}` and it saturates at 1 for the small classes.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        if p >= 1.0 {
            true
        } else if p <= 0.0 {
            false
        } else {
            self.next_f64() < p
        }
    }

    /// Uniform integer in `[0, bound)`. `bound == 0` returns 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        // Lemire-style rejection, so the result is exactly uniform.
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % bound;
            }
        }
    }

    /// Derive `count` independent child streams.
    ///
    /// Each vertex of the simulated network gets its own stream, so a vertex's
    /// coin flips do not depend on the order in which the simulator visits
    /// vertices.
    pub fn split(&mut self, count: usize) -> Vec<SplitMix64> {
        (0..count).map(|_| SplitMix64::seed(self.next_u64())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::seed(12345);
        let mut b = SplitMix64::seed(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::seed(1);
        let mut b = SplitMix64::seed(2);
        let differ = (0..64).filter(|_| a.next_u64() != b.next_u64()).count();
        assert_eq!(differ, 64);
    }

    #[test]
    fn floats_stay_in_range() {
        let mut r = SplitMix64::seed(7);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn bernoulli_is_roughly_calibrated() {
        let mut r = SplitMix64::seed(99);
        let hits = (0..20_000).filter(|_| r.bernoulli(0.25)).count();
        // 20k draws at p = 0.25 has standard deviation about 61.
        assert!((4700..5300).contains(&hits), "hits = {hits}");
        assert!((0..1000).all(|_| r.bernoulli(1.0)));
        assert!(!(0..1000).any(|_| r.bernoulli(0.0)));
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = SplitMix64::seed(4242);
        for _ in 0..5000 {
            assert!(r.below(37) < 37);
        }
        assert_eq!(r.below(1), 0);
        assert_eq!(r.below(0), 0);
    }
}
