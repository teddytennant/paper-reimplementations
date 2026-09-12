//! Sparse multivariate polynomials over F_p, plus dense univariates in y.

use crate::field::{self, P};
use std::collections::BTreeMap;

/// Sparse term: exponents of x_1..x_n followed by the y-exponent, plus coeff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub xexp: Vec<u32>,
    pub yexp: u32,
    pub coeff: u64,
}

/// Sparse polynomial in F_p[x_1,...,x_n, y].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparsePoly {
    pub n: usize,
    pub terms: Vec<Term>,
}

impl SparsePoly {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            terms: Vec::new(),
        }
    }

    pub fn from_terms(n: usize, terms: Vec<Term>) -> Self {
        let mut p = Self { n, terms };
        p.normalize();
        p
    }

    pub fn monomial(n: usize, xexp: Vec<u32>, yexp: u32, coeff: u64) -> Self {
        assert_eq!(xexp.len(), n);
        Self::from_terms(
            n,
            vec![Term {
                xexp,
                yexp,
                coeff: coeff % P,
            }],
        )
    }

    pub fn constant(n: usize, c: u64) -> Self {
        if c % P == 0 {
            Self::new(n)
        } else {
            Self::from_terms(
                n,
                vec![Term {
                    xexp: vec![0; n],
                    yexp: 0,
                    coeff: c % P,
                }],
            )
        }
    }

    fn key(t: &Term) -> (Vec<u32>, u32) {
        (t.xexp.clone(), t.yexp)
    }

    pub fn normalize(&mut self) {
        let mut map: BTreeMap<(Vec<u32>, u32), u64> = BTreeMap::new();
        for t in self.terms.drain(..) {
            assert_eq!(t.xexp.len(), self.n);
            let k = Self::key(&t);
            let e = map.entry(k).or_insert(0);
            *e = field::add(*e, t.coeff % P);
        }
        self.terms = map
            .into_iter()
            .filter(|(_, c)| *c != 0)
            .map(|((xexp, yexp), coeff)| Term { xexp, yexp, coeff })
            .collect();
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn total_degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|t| t.xexp.iter().sum::<u32>() + t.yexp)
            .max()
            .unwrap_or(0)
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        Self::from_terms(self.n, terms)
    }

    pub fn scale(&self, c: u64) -> Self {
        if c % P == 0 {
            return Self::new(self.n);
        }
        let terms = self
            .terms
            .iter()
            .map(|t| Term {
                xexp: t.xexp.clone(),
                yexp: t.yexp,
                coeff: field::mul(t.coeff, c),
            })
            .collect();
        Self::from_terms(self.n, terms)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(field::sub(0, 1)))
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut terms = Vec::new();
        for a in &self.terms {
            for b in &other.terms {
                let xexp: Vec<u32> = a
                    .xexp
                    .iter()
                    .zip(b.xexp.iter())
                    .map(|(u, v)| u + v)
                    .collect();
                terms.push(Term {
                    xexp,
                    yexp: a.yexp + b.yexp,
                    coeff: field::mul(a.coeff, b.coeff),
                });
            }
        }
        Self::from_terms(self.n, terms)
    }

    /// Evaluate at x = b, leaving a univariate in y (dense coeffs).
    pub fn eval_x(&self, b: &[u64]) -> UniPoly {
        assert_eq!(b.len(), self.n);
        let mut max_y = 0u32;
        for t in &self.terms {
            max_y = max_y.max(t.yexp);
        }
        let mut coeffs = vec![0u64; max_y as usize + 1];
        for t in &self.terms {
            let mut v = t.coeff;
            for (k, &e) in t.xexp.iter().enumerate() {
                v = field::mul(v, field::pow(b[k], e as u64));
            }
            let i = t.yexp as usize;
            coeffs[i] = field::add(coeffs[i], v);
        }
        UniPoly::from_coeffs(coeffs)
    }

    /// Partial derivative wrt x_k (Lemma 18 uses the sum of these).
    pub fn partial_x(&self, k: usize) -> Self {
        assert!(k < self.n);
        let mut terms = Vec::new();
        for t in &self.terms {
            let e = t.xexp[k];
            if e == 0 {
                continue;
            }
            let mut xexp = t.xexp.clone();
            xexp[k] = e - 1;
            terms.push(Term {
                xexp,
                yexp: t.yexp,
                coeff: field::mul(t.coeff, (e as u64) % P),
            });
        }
        Self::from_terms(self.n, terms)
    }

    /// S(g) = sum_k ∂g/∂x_k (Lemma 18).
    pub fn sum_partials(&self) -> Self {
        let mut acc = Self::new(self.n);
        for k in 0..self.n {
            acc = acc.add(&self.partial_x(k));
        }
        acc
    }

    /// Substitute y = 1, returning a poly in x only (y-exponent folded away).
    pub fn set_y_one(&self) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|t| Term {
                xexp: t.xexp.clone(),
                yexp: 0,
                coeff: t.coeff,
            })
            .collect();
        Self::from_terms(self.n, terms)
    }

    /// Make the leading y-coefficient (as a poly in x) monic at a point,
    /// used after recovery. Here we just scale so the highest-y term's
    /// coefficient is 1 if it is a constant.
    pub fn make_lc_one_if_const(&self) -> Self {
        let max_y = self.terms.iter().map(|t| t.yexp).max().unwrap_or(0);
        let lc_terms: Vec<&Term> = self.terms.iter().filter(|t| t.yexp == max_y).collect();
        if lc_terms.len() == 1 && lc_terms[0].xexp.iter().all(|&e| e == 0) {
            let inv = field::inv(lc_terms[0].coeff);
            self.scale(inv)
        } else {
            self.clone()
        }
    }
}

/// Dense univariate polynomial in F_p[y], low degree first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniPoly {
    pub coeffs: Vec<u64>,
}

impl UniPoly {
    pub fn zero() -> Self {
        Self { coeffs: vec![0] }
    }

    pub fn from_coeffs(mut coeffs: Vec<u64>) -> Self {
        for c in &mut coeffs {
            *c %= P;
        }
        while coeffs.len() > 1 && *coeffs.last().unwrap() == 0 {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(0);
        }
        Self { coeffs }
    }

    pub fn deg(&self) -> i32 {
        if self.coeffs.len() == 1 && self.coeffs[0] == 0 {
            -1
        } else {
            self.coeffs.len() as i32 - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.deg() < 0
    }

    pub fn lc(&self) -> u64 {
        *self.coeffs.last().unwrap_or(&0)
    }

    pub fn monic(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        self.scale(field::inv(self.lc()))
    }

    pub fn scale(&self, c: u64) -> Self {
        Self::from_coeffs(self.coeffs.iter().map(|&a| field::mul(a, c)).collect())
    }

    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0u64; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            out[i] = field::add(out[i], a);
        }
        for (i, &a) in other.coeffs.iter().enumerate() {
            out[i] = field::add(out[i], a);
        }
        Self::from_coeffs(out)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(field::sub(0, 1)))
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0u64; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] = field::add(out[i + j], field::mul(a, b));
            }
        }
        Self::from_coeffs(out)
    }

    /// Polynomial division: (q, r) with self = q * div + r, deg r < deg div.
    pub fn div_rem(&self, div: &Self) -> (Self, Self) {
        assert!(!div.is_zero(), "division by zero polynomial");
        let mut rem = self.coeffs.clone();
        if self.deg() < div.deg() {
            return (Self::zero(), self.clone());
        }
        let mut q = vec![0u64; (self.deg() - div.deg() + 1) as usize];
        let inv_lc = field::inv(div.lc());
        while rem.len() > 1 && *rem.last().unwrap() == 0 {
            rem.pop();
        }
        while (rem.len() as i32 - 1) >= div.deg() && !(rem.len() == 1 && rem[0] == 0) {
            let d = rem.len() - div.coeffs.len();
            let coef = field::mul(*rem.last().unwrap(), inv_lc);
            q[d] = coef;
            for (i, &a) in div.coeffs.iter().enumerate() {
                let idx = i + d;
                rem[idx] = field::sub(rem[idx], field::mul(a, coef));
            }
            while rem.len() > 1 && *rem.last().unwrap() == 0 {
                rem.pop();
            }
        }
        (Self::from_coeffs(q), Self::from_coeffs(rem))
    }

    pub fn quo(&self, div: &Self) -> Self {
        self.div_rem(div).0
    }

    pub fn rem(&self, div: &Self) -> Self {
        self.div_rem(div).1
    }

    pub fn gcd(a: &Self, b: &Self) -> Self {
        let mut r0 = a.clone();
        let mut r1 = b.clone();
        while !r1.is_zero() {
            let r = r0.rem(&r1);
            r0 = r1;
            r1 = r;
        }
        r0.monic()
    }

    /// Extended Euclidean: u g + v h = gcd (monic gcd, u,v scaled accordingly).
    /// Returns (g, u, v) with g monic and u*a + v*b = g.
    pub fn xgcd(a: &Self, b: &Self) -> (Self, Self, Self) {
        let mut r0 = a.clone();
        let mut r1 = b.clone();
        let mut s0 = UniPoly::from_coeffs(vec![1]);
        let mut s1 = UniPoly::zero();
        let mut t0 = UniPoly::zero();
        let mut t1 = UniPoly::from_coeffs(vec![1]);
        while !r1.is_zero() {
            let (q, r) = r0.div_rem(&r1);
            let s = s0.sub(&q.mul(&s1));
            let t = t0.sub(&q.mul(&t1));
            r0 = r1;
            r1 = r;
            s0 = s1;
            s1 = s;
            t0 = t1;
            t1 = t;
        }
        if r0.is_zero() {
            return (Self::zero(), s0, t0);
        }
        let inv = field::inv(r0.lc());
        (r0.scale(inv), s0.scale(inv), t0.scale(inv))
    }

    /// Coefficient of y^d, or 0.
    pub fn coeff(&self, d: u32) -> u64 {
        self.coeffs.get(d as usize).copied().unwrap_or(0)
    }

    pub fn support_degrees(&self) -> Vec<u32> {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(i, _)| i as u32)
            .collect()
    }
}
