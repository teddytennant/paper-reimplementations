use sparse_poly_gcd::factor::factor_partial;
use sparse_poly_gcd::field;
use sparse_poly_gcd::gcd::field_gcd;
use sparse_poly_gcd::hensel::{one_step_lift, Dual};
use sparse_poly_gcd::poly::{SparsePoly, Term, UniPoly};
use sparse_poly_gcd::recover::recover;

fn term(xexp: Vec<u32>, yexp: u32, coeff: u64) -> Term {
    Term { xexp, yexp, coeff }
}

/// Algorithm 1 recovers a known sparse y-separated poly at b = (3, 5).
#[test]
fn alg1_recovers_y_separated() {
    let n = 2;
    let f = SparsePoly::from_terms(
        n,
        vec![
            term(vec![2, 1], 3, 4),
            term(vec![0, 2], 1, 9),
            term(vec![1, 0], 0, 2),
        ],
    );
    let b = [3u64, 5];
    let f_b = f.eval_x(&b);
    let partials: Vec<UniPoly> = (0..n).map(|k| f.partial_x(k).eval_x(&b)).collect();
    let rec = recover(&b, &f_b, &partials, f.total_degree()).unwrap();
    assert_eq!(rec, f);
}

/// Algorithm 2: F = G H, start from (G0, H0) = (G, H) mod z, lift, check
/// F ≡ G1 H1 (mod z^2) and G1 ≡ G0 (mod z).
#[test]
fn alg2_one_step_hensel() {
    // Work in K[z, y]/(z^2) with G, H univariates that pick up a z term
    // from the substitution x_i = z + b_i. Use a constructed dual:
    // G = (y + 1) + z (2y), H = (y + 3) + z (y + 4), F = G H.
    let g0 = Dual {
        c0: UniPoly::from_coeffs(vec![1, 1]), // 1 + y
        c1: UniPoly::from_coeffs(vec![0, 2]), // 2y
    };
    let h0 = Dual {
        c0: UniPoly::from_coeffs(vec![3, 1]), // 3 + y
        c1: UniPoly::from_coeffs(vec![4, 1]), // 4 + y
    };
    let f = g0.mul(&h0);
    // Bézout of the z^0 parts: u G0 + v H0 = 1.
    let (_g, u0, v0) = UniPoly::xgcd(&g0.c0, &h0.c0);
    assert_eq!(_g.deg(), 0);
    let u = Dual::from_const(u0);
    let v = Dual::from_const(v0);
    // Start the lift from the z^0 projections, as Algorithm 2 does.
    let g_start = Dual::from_const(g0.c0.clone());
    let h_start = Dual::from_const(h0.c0.clone());
    let (g1, h1) = one_step_lift(&f, &g_start, &h_start, &u, &v);
    // G1 ≡ G0 (mod z)
    assert_eq!(g1.c0, g0.c0);
    // F ≡ G1 H1 (mod z^2)
    let prod = g1.mul(&h1);
    assert_eq!(prod.c0, f.c0);
    assert_eq!(prod.c1, f.c1);
}

/// Expand g(b1+z, ..., bn+z, y) as Dual (const + z * zcoeff) in y.
fn eval_shift_dual(g: &SparsePoly, b: &[u64]) -> Dual {
    let n = g.n;
    let mut c0 = UniPoly::zero();
    let mut c1 = UniPoly::zero();
    for t in &g.terms {
        let mut prod = t.coeff;
        for k in 0..n {
            prod = field::mul(prod, field::pow(b[k], t.xexp[k] as u64));
        }
        let mut zpart = 0u64;
        for k in 0..n {
            let e = t.xexp[k];
            if e == 0 {
                continue;
            }
            let mut v = t.coeff;
            v = field::mul(v, (e as u64) % field::P);
            v = field::mul(v, field::pow(b[k], (e - 1) as u64));
            for j in 0..n {
                if j != k {
                    v = field::mul(v, field::pow(b[j], t.xexp[j] as u64));
                }
            }
            zpart = field::add(zpart, v);
        }
        let mut d0 = vec![0u64; t.yexp as usize + 1];
        d0[t.yexp as usize] = prod;
        let mut d1 = vec![0u64; t.yexp as usize + 1];
        d1[t.yexp as usize] = zpart;
        c0 = c0.add(&UniPoly::from_coeffs(d0));
        c1 = c1.add(&UniPoly::from_coeffs(d1));
    }
    Dual { c0, c1 }
}

/// Lemma 19: g(z+b1, ..., z+bn, y) ≡ g(b,y) + S(g)(b,y) z  (mod z^2).
#[test]
fn lemma19_shift_identity() {
    let n = 2;
    let g = SparsePoly::from_terms(
        n,
        vec![
            term(vec![2, 0], 1, 3),
            term(vec![0, 1], 2, 5),
            term(vec![1, 1], 0, 7),
        ],
    );
    let b = [3u64, 4];
    let s = g.sum_partials();
    let dual = eval_shift_dual(&g, &b);
    assert_eq!(dual.c0, g.eval_x(&b));
    assert_eq!(dual.c1, s.eval_x(&b));
    let s2 = g.partial_x(0).add(&g.partial_x(1));
    assert_eq!(s, s2);
}

/// Theorem 24 matches analytic partials of constructed factors.
#[test]
fn thm24_matches_analytic_partials() {
    let n = 2;
    let g = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 0], 1, 1), // y  (monic in y)
            term(vec![1, 0], 0, 2), // 2 x1
        ],
    );
    let h = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 1], 1, 1), // x2 y
            term(vec![0, 0], 0, 3), // 3
        ],
    );
    let f = g.mul(&h);
    let b = [4u64, 5];
    let g_b = g.eval_x(&b);
    let h_b = h.eval_x(&b);
    let (_gcd, u, v) = UniPoly::xgcd(&g_b, &h_b);
    assert_eq!(_gcd.deg(), 0);
    for k in 0..n {
        let df = f.partial_x(k).eval_x(&b);
        let (dg_hat, dh_hat) = factor_partial(&df, &g_b, &h_b, &u, &v);
        let dg = g.partial_x(k).eval_x(&b);
        let dh = h.partial_x(k).eval_x(&b);
        assert_eq!(dg_hat, dg, "∂g/∂x_{k} mismatch");
        assert_eq!(dh_hat, dh, "∂h/∂x_{k} mismatch");
    }
}

fn gcd_recovers(n: usize, g: SparsePoly, a1: SparsePoly, b1: SparsePoly, eval: &[u64]) {
    let a = g.mul(&a1);
    let b = g.mul(&b1);
    let rec = field_gcd(&a, &b, eval).expect("field gcd");
    // Recovered G is monic in y (G0 was monic), so scale the constructed G
    // the same way.
    let g_b = g.eval_x(eval);
    let rec_b = rec.eval_x(eval);
    assert_eq!(rec_b.monic(), g_b.monic());
    // Same support after making leading y-coeff 1 at b via Algorithm 1.
    // Compare polynomials after scaling rec to match g's leading coeff.
    let g_monic = {
        let gb = g.eval_x(eval);
        if gb.is_zero() {
            g.clone()
        } else {
            g.scale(field::inv(gb.lc()))
        }
    };
    assert_eq!(rec, g_monic);
    let _ = n;
}

/// gcd(G A1, G B1) recovers G for n=2.
/// G is monic in y and y-separated (Algorithm 1 needs distinct y-degrees).
#[test]
fn gcd_n2() {
    let n = 2;
    let g = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 0], 2, 1), // y^2
            term(vec![1, 0], 1, 3), // 3 x1 y
            term(vec![0, 1], 0, 5), // 5 x2
        ],
    );
    let a1 = SparsePoly::from_terms(
        n,
        vec![
            term(vec![1, 0], 0, 1),
            term(vec![0, 0], 0, 7),
        ],
    );
    let b1 = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 1], 0, 1),
            term(vec![0, 0], 0, 11),
        ],
    );
    gcd_recovers(n, g, a1, b1, &[3, 5]);
}

/// gcd(G A1, G B1) recovers G for n=3.
#[test]
fn gcd_n3() {
    let n = 3;
    let g = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 0, 0], 3, 1), // y^3
            term(vec![1, 0, 0], 2, 2), // 2 x1 y^2
            term(vec![0, 1, 0], 1, 3), // 3 x2 y
            term(vec![0, 0, 1], 0, 4), // 4 x3
        ],
    );
    let a1 = SparsePoly::from_terms(
        n,
        vec![
            term(vec![1, 0, 0], 0, 1),
            term(vec![0, 0, 0], 0, 5),
        ],
    );
    let b1 = SparsePoly::from_terms(
        n,
        vec![
            term(vec![0, 1, 0], 0, 1),
            term(vec![0, 0, 1], 0, 1),
            term(vec![0, 0, 0], 0, 6),
        ],
    );
    gcd_recovers(n, g, a1, b1, &[3, 5, 7]);
}
