//! Prime field F_p with p = 1_000_000_007, large enough that char > deg
//! for the small polynomials used here (Huang & Gao, arXiv:2609.08074).

pub const P: u64 = 1_000_000_007;

#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}

#[inline]
pub fn sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + P - b
    }
}

#[inline]
pub fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

pub fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    base %= P;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul(acc, base);
        }
        base = mul(base, base);
        exp >>= 1;
    }
    acc
}

/// Modular inverse via extended Euclidean algorithm. Panics if a ≡ 0.
pub fn inv(a: u64) -> u64 {
    let a = (a % P) as i128;
    if a == 0 {
        panic!("inverse of 0 in F_p");
    }
    let mut t = 0i128;
    let mut newt = 1i128;
    let mut r = P as i128;
    let mut newr = a;
    while newr != 0 {
        let q = r / newr;
        let tmp = newt;
        newt = t - q * newt;
        t = tmp;
        let tmp = newr;
        newr = r - q * newr;
        r = tmp;
    }
    if r > 1 {
        panic!("not invertible");
    }
    if t < 0 {
        t += P as i128;
    }
    t as u64
}

pub fn from_i64(x: i64) -> u64 {
    let p = P as i64;
    let mut r = x % p;
    if r < 0 {
        r += p;
    }
    r as u64
}

/// Map a field element that is known to be a small integer exponent
/// in 0..=max_deg back to u32. The paper requires char(K) > deg so this
/// identification is unique.
pub fn as_exponent(x: u64, max_deg: u32) -> Result<u32, String> {
    if x <= max_deg as u64 {
        Ok(x as u32)
    } else {
        Err(format!(
            "field element {x} is not an exponent in 0..={max_deg}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_roundtrip() {
        for a in [1u64, 2, 3, 7, 11, 1234567] {
            assert_eq!(mul(a, inv(a)), 1);
        }
    }
}
