use zonotope_tv::{approx_tv_product, exact_tv_product, ProductMixture};

fn main() {
    let n = 3usize;
    let eps = 0.5;
    let p = ProductMixture {
        weights: vec![0.4, 0.6],
        marginals: vec![
            vec![vec![0.2, 0.8], vec![0.5, 0.5], vec![0.9, 0.1]],
            vec![vec![0.7, 0.3], vec![0.4, 0.6], vec![0.2, 0.8]],
        ],
    };
    let q = ProductMixture {
        weights: vec![1.0],
        marginals: vec![vec![
            vec![0.5, 0.5],
            vec![0.3, 0.7],
            vec![0.6, 0.4],
        ]],
    };
    let exact = exact_tv_product(&p, &q);
    let approx = approx_tv_product(&p, &q, eps);
    let eta = (1.0 + eps).powf(1.0 / (n as f64)) - 1.0;
    let ratio = exact / approx;
    println!("n=3 q=2 k1=2 k2=1 eps=0.5");
    println!("exact_tv={exact:.12}");
    println!("approx_tv={approx:.12}");
    println!("ratio_exact_over_approx={ratio:.12}");
    println!("eta={eta:.12}");
}
