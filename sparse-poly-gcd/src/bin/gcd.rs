use sparse_poly_gcd::{field_gcd, SparsePoly, Term};

fn main() {
    // G = y^2 + 3 x1 y + 5 x2 (monic in y, y-separated),
    // A1 = x1 + 7, B1 = x2 + 11.
    let n = 2;
    let g = SparsePoly::from_terms(
        n,
        vec![
            Term {
                xexp: vec![0, 0],
                yexp: 2,
                coeff: 1,
            },
            Term {
                xexp: vec![1, 0],
                yexp: 1,
                coeff: 3,
            },
            Term {
                xexp: vec![0, 1],
                yexp: 0,
                coeff: 5,
            },
        ],
    );
    let a1 = SparsePoly::from_terms(
        n,
        vec![
            Term {
                xexp: vec![1, 0],
                yexp: 0,
                coeff: 1,
            },
            Term {
                xexp: vec![0, 0],
                yexp: 0,
                coeff: 7,
            },
        ],
    );
    let b1 = SparsePoly::from_terms(
        n,
        vec![
            Term {
                xexp: vec![0, 1],
                yexp: 0,
                coeff: 1,
            },
            Term {
                xexp: vec![0, 0],
                yexp: 0,
                coeff: 11,
            },
        ],
    );
    let a = g.mul(&a1);
    let b = g.mul(&b1);
    let eval = [3u64, 5];
    let rec = field_gcd(&a, &b, &eval).expect("gcd");
    println!("recovered G ({} terms):", rec.terms.len());
    for t in &rec.terms {
        println!(
            "  coeff={} x^{:?} y^{}",
            t.coeff, t.xexp, t.yexp
        );
    }
}
