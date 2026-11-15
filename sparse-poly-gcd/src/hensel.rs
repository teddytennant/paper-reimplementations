//! Algorithm 2: one-step Hensel lifting modulo z^2.

use crate::poly::UniPoly;

/// Element of F_p[z, y] / (z^2): const + z * zcoeff, each in F_p[y].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dual {
    pub c0: UniPoly,
    pub c1: UniPoly,
}

impl Dual {
    pub fn from_const(p: UniPoly) -> Self {
        Self {
            c0: p,
            c1: UniPoly::zero(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        // (a0 + a1 z)(b0 + b1 z) = a0 b0 + (a0 b1 + a1 b0) z  (mod z^2)
        Self {
            c0: self.c0.mul(&other.c0),
            c1: self.c0.mul(&other.c1).add(&self.c1.mul(&other.c0)),
        }
    }
}

/// Algorithm 2. F, G0, H0, Bézout u, v with u G0 + v H0 = 1, G0 monic.
/// All of F, G0, H0, u, v may carry a z^1 term (they live in K[z,y]/(z^2)).
///
/// e ≡ F − G0 H0 (mod z^2)
/// v e ≡ q G0 + r (mod z^2), deg_y r < deg_y G0
/// H1 ≡ H0 + u e + q H0 (mod z^2)
/// G1 ≡ G0 + r (mod z^2)
pub fn one_step_lift(f: &Dual, g0: &Dual, h0: &Dual, u: &Dual, v: &Dual) -> (Dual, Dual) {
    let e = f.sub(&g0.mul(h0));
    let ve = v.mul(&e);
    // Divide in y, coefficient-wise in z. deg_y G0 is taken from the z^0 part
    // (G0 is monic in y, so its leading term does not involve z).
    let g0_y = &g0.c0;
    let (q0, r0) = ve.c0.div_rem(g0_y);
    let (q1, r1) = ve.c1.div_rem(g0_y);
    let q = Dual { c0: q0, c1: q1 };
    let r = Dual { c0: r0, c1: r1 };
    let h1 = h0.add(&u.mul(&e)).add(&q.mul(h0));
    let g1 = g0.add(&r);
    (g1, h1)
}
