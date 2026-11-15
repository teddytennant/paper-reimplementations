//! Prime field GF(p) with i64 representatives in `0..p`.

use std::fmt;

/// A prime field `GF(p)`. The constructor does not prove primality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp {
    p: i64,
}

impl Fp {
    /// GF(2).
    pub const GF2: Fp = Fp { p: 2 };
    /// A 15-bit prime used in F4-style linear algebra.
    pub const P32003: Fp = Fp { p: 32003 };
    /// Largest prime less than 2^16.
    pub const P65521: Fp = Fp { p: 65521 };

    pub fn new(p: i64) -> Self {
        assert!(p >= 2, "modulus must be at least 2");
        Fp { p }
    }

    #[inline]
    pub fn modulus(self) -> i64 {
        self.p
    }

    /// Reduce an arbitrary integer into `0..p`.
    #[inline]
    pub fn reduce(self, a: i64) -> i64 {
        let r = a % self.p;
        if r < 0 {
            r + self.p
        } else {
            r
        }
    }

    #[inline]
    pub fn add(self, a: i64, b: i64) -> i64 {
        let s = a + b;
        if s >= self.p {
            s - self.p
        } else {
            s
        }
    }

    #[inline]
    pub fn sub(self, a: i64, b: i64) -> i64 {
        if a >= b {
            a - b
        } else {
            a + self.p - b
        }
    }

    #[inline]
    pub fn neg(self, a: i64) -> i64 {
        if a == 0 {
            0
        } else {
            self.p - a
        }
    }

    #[inline]
    pub fn mul(self, a: i64, b: i64) -> i64 {
        (a as i128 * b as i128).rem_euclid(self.p as i128) as i64
    }

    /// Modular inverse via extended Euclid. Panics if `a == 0`.
    pub fn inv(self, a: i64) -> i64 {
        assert!(a != 0, "inverse of 0 in GF(p)");
        let (g, x, _) = egcd(a, self.p);
        assert!(g == 1 || g == -1, "element is not invertible");
        self.reduce(x)
    }
}

fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = egcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GF({})", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf2_arithmetic() {
        let f = Fp::GF2;
        assert_eq!(f.add(1, 1), 0);
        assert_eq!(f.sub(0, 1), 1);
        assert_eq!(f.mul(1, 1), 1);
        assert_eq!(f.inv(1), 1);
        assert_eq!(f.neg(1), 1);
    }

    #[test]
    fn gf7_inv_roundtrip() {
        let f = Fp::new(7);
        for a in 1..7 {
            assert_eq!(f.mul(a, f.inv(a)), 1);
        }
        assert_eq!(f.add(3, 5), 1);
        assert_eq!(f.sub(3, 5), 5);
        assert_eq!(f.reduce(-1), 6);
    }

    #[test]
    fn larger_primes_mul_fits() {
        for f in [Fp::P32003, Fp::P65521] {
            let a = f.modulus() - 1;
            assert_eq!(f.mul(a, a), 1);
            assert_eq!(f.mul(a, f.inv(a)), 1);
        }
    }
}
