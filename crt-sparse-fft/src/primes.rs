//! Stage-prime selection for the engineered exact-product family (Def. II.2).

/// Trial-division primality. Stage primes are Θ(k) and small.
pub fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut d: usize = 3;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Primes in the closed interval `[lo, hi]`.
pub fn primes_in_range(lo: usize, hi: usize) -> Vec<usize> {
    (lo..=hi).filter(|&p| is_prime(p)).collect()
}

/// Product of `primes`, or `None` on overflow.
pub fn product(primes: &[usize]) -> Option<usize> {
    primes.iter().try_fold(1usize, |acc, &p| acc.checked_mul(p))
}

/// True iff `n` is the product of `L >= 2` pairwise-distinct odd primes.
pub fn is_engineered_length(n: usize, primes: &[usize]) -> bool {
    if primes.len() < 2 {
        return false;
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut prod = 1usize;
    for &p in primes {
        if p < 3 || !is_prime(p) || !seen.insert(p) {
            return false;
        }
        prod = match prod.checked_mul(p) {
            Some(v) => v,
            None => return false,
        };
    }
    prod == n
}

/// Smallest integer `c >= 2` such that every prime lies in `[k, c k]`.
pub fn envelope_c(k: usize, primes: &[usize]) -> usize {
    let mut c = 2usize;
    if k == 0 {
        return c;
    }
    for &p in primes {
        let need = p.div_ceil(k);
        if need > c {
            c = need;
        }
    }
    c
}

/// Pick `l` pairwise-distinct primes in `[k, c k]`, widening `c` from 2
/// if `[k, 2k]` has too few primes (Def. II.2).
///
/// Returns `None` if `k < 3`, `l < 2`, or no such list exists under a
/// modest widening cap.
pub fn engineered_primes(k: usize, l: usize) -> Option<Vec<usize>> {
    if k < 3 || l < 2 {
        return None;
    }
    let mut c = 2usize;
    loop {
        let hi = k.saturating_mul(c);
        let primes = primes_in_range(k, hi);
        if primes.len() >= l {
            return Some(primes.into_iter().take(l).collect());
        }
        if c >= 64 || hi >= 10_000 {
            return None;
        }
        c += 1;
    }
}
