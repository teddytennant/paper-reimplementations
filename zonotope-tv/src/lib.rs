//! Deterministic relative approximation of total variation between mixtures
//! of product distributions, by zonotope compression.
//!
//! Theorem 1.2 of Fu, arXiv:2609.24235. Markov-chain mixtures and latent-tree
//! Ising models are not implemented.
//!
//! Bad inputs panic: weights or marginals that are negative or do not sum to 1
//! (absolute tolerance 1e-8), eps outside (0, 1), or a shape mismatch between
//! the two mixtures.

mod compress;
mod linalg;

pub use compress::compress;

use linalg::dot;

/// Mixture of product distributions on [q]^n.
///
/// `weights[s]` is the mixing weight of component s.
/// `marginals[s][i][v]` is the probability of symbol v on coordinate i
/// under component s. Every mixture passed to the TV routines must use the
/// same n and the same q.
#[derive(Clone, Debug)]
pub struct ProductMixture {
    pub weights: Vec<f64>,
    pub marginals: Vec<Vec<Vec<f64>>>,
}

/// Support function of a represented zonotope: sum_i |dot(g_i, direction)|.
pub fn support(generators: &[Vec<f64>], direction: &[f64]) -> f64 {
    let mut total = 0.0;
    for g in generators {
        if g.len() != direction.len() {
            panic!("generator dimension does not match the direction");
        }
        total += dot(g, direction).abs();
    }
    total
}

/// Exact TV distance. Enumerates every atom of [q]^n.
pub fn exact_tv_product(p: &ProductMixture, q: &ProductMixture) -> f64 {
    let (n, q_size) = validate_pair(p, q, None);
    let mut acc = 0.0;
    for_each_atom(n, q_size, &mut |x| {
        let diff = mixture_prob(p, x) - mixture_prob(q, x);
        acc += diff.abs();
    });
    0.5 * acc
}

/// Relative approximation of TV distance from Algorithm 1, specialized to
/// mixtures of product distributions (n diagonal-sum gates).
///
/// Guarantee, up to floating-point error: d_hat <= d_TV <= (1 + eps) d_hat.
pub fn approx_tv_product(p: &ProductMixture, q: &ProductMixture, eps: f64) -> f64 {
    let (n, q_size) = validate_pair(p, q, Some(eps));
    let k1 = p.weights.len();
    let k2 = q.weights.len();
    let k = k1 + k2;
    let eta = (1.0 + eps).powf(1.0 / (n as f64)) - 1.0;
    let mut gamma = Vec::with_capacity(k);
    for &alpha in &p.weights {
        gamma.push(alpha);
    }
    for &beta in &q.weights {
        gamma.push(-beta);
    }
    let mut gens = vec![vec![1.0; k]];
    for i in 0..n {
        let mut next = Vec::with_capacity(gens.len() * q_size);
        for v in 0..q_size {
            for g in &gens {
                let mut scaled = vec![0.0; k];
                for a in 0..k {
                    let prob = if a < k1 {
                        p.marginals[a][i][v]
                    } else {
                        q.marginals[a - k1][i][v]
                    };
                    scaled[a] = g[a] * prob;
                }
                next.push(scaled);
            }
        }
        gens = compress(&next, eta);
    }
    0.5 * support(&gens, &gamma)
}

fn mixture_prob(dist: &ProductMixture, atom: &[usize]) -> f64 {
    let mut total = 0.0;
    for (s, &weight) in dist.weights.iter().enumerate() {
        let mut prod = weight;
        for (i, &symbol) in atom.iter().enumerate() {
            prod *= dist.marginals[s][i][symbol];
        }
        total += prod;
    }
    total
}

fn for_each_atom(n: usize, q_size: usize, visit: &mut dyn FnMut(&[usize])) {
    let mut atom = vec![0usize; n];
    loop {
        visit(&atom);
        let mut i = 0;
        loop {
            if i == n {
                return;
            }
            atom[i] += 1;
            if atom[i] < q_size {
                break;
            }
            atom[i] = 0;
            i += 1;
        }
    }
}

fn validate_pair(p: &ProductMixture, q: &ProductMixture, eps: Option<f64>) -> (usize, usize) {
    if let Some(eps) = eps {
        if !(eps > 0.0 && eps < 1.0) {
            panic!("eps must lie in (0, 1)");
        }
    }
    let (n, q_size) = validate_one(p);
    let (n2, q2) = validate_one(q);
    if n != n2 || q_size != q2 {
        panic!("P and Q must share n and q");
    }
    (n, q_size)
}

fn validate_one(dist: &ProductMixture) -> (usize, usize) {
    if dist.weights.is_empty() || dist.marginals.is_empty() {
        panic!("a mixture needs at least one component");
    }
    if dist.weights.len() != dist.marginals.len() {
        panic!("weight count does not match the number of components");
    }
    let n = dist.marginals[0].len();
    if n == 0 {
        panic!("a mixture needs at least one coordinate");
    }
    let q_size = dist.marginals[0][0].len();
    if q_size == 0 {
        panic!("a mixture needs at least one symbol");
    }
    let mut weight_sum = 0.0;
    for &w in &dist.weights {
        if !w.is_finite() || w < 0.0 {
            panic!("mixing weights must be nonnegative and finite");
        }
        weight_sum += w;
    }
    if (weight_sum - 1.0).abs() > 1e-8 {
        panic!("mixing weights must sum to 1");
    }
    for component in &dist.marginals {
        if component.len() != n {
            panic!("every component must have n coordinates");
        }
        for marginal in component {
            if marginal.len() != q_size {
                panic!("every marginal must have q symbols");
            }
            let mut sum = 0.0;
            for &prob in marginal {
                if !prob.is_finite() || prob < 0.0 {
                    panic!("marginals must be nonnegative and finite");
                }
                sum += prob;
            }
            if (sum - 1.0).abs() > 1e-8 {
                panic!("each marginal must sum to 1");
            }
        }
    }
    (n, q_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::{
        best_net_cosine_brute, best_net_cosine_cone, lewis_gram_quads, lewis_iteration_count,
        lewis_weights,
    };
    use crate::linalg::{sphere_constant, symmetric_eigen};

    fn mix(weights: Vec<f64>, marginals: Vec<Vec<Vec<f64>>>) -> ProductMixture {
        ProductMixture { weights, marginals }
    }

    #[test]
    fn sphere_constants_are_tight() {
        assert_eq!(sphere_constant(1), 1.0);
        assert_eq!(sphere_constant(2), 2.0 / std::f64::consts::PI);
        assert!((sphere_constant(3) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn support_matches_absolute_inner_products() {
        let gens = vec![
            vec![1.0, 0.0],
            vec![0.0, 2.0],
            vec![3.0, -1.0],
        ];
        let direction = vec![0.5, -0.25];
        let expected = (1.0_f64 * 0.5 + 0.0 * -0.25).abs()
            + (0.0_f64 * 0.5 + 2.0 * -0.25).abs()
            + (3.0_f64 * 0.5 + -1.0 * -0.25).abs();
        assert_eq!(support(&gens, &direction), expected);
        assert_eq!(expected, 2.75);
    }

    #[test]
    fn jacobi_recovers_a_known_spectrum() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (evals, evecs) = symmetric_eigen(&a);
        assert!((evals[0] - 3.0).abs() < 1e-10);
        assert!((evals[1] - 1.0).abs() < 1e-10);
        let mut recon = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    recon[i][j] += evecs[i][k] * evals[k] * evecs[j][k];
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!((recon[i][j] - a[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn lewis_weights_fixed_point_and_paper_factor() {
        assert_eq!(lewis_weights(&[vec![0.4, -0.2, 0.7]], 5), vec![1.0]);
        let rows = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
        ];
        let long = lewis_weights(&rows, 40);
        let quads = lewis_gram_quads(&rows, &long);
        for i in 0..rows.len() {
            let rel = (long[i] * long[i] - quads[i]).abs() / (long[i] * long[i]);
            assert!(
                rel < 1e-6,
                "Lewis fixed point failed at {i}: w^2={} quad={} rel={rel}",
                long[i] * long[i],
                quads[i]
            );
        }
        let t = lewis_iteration_count(rows.len());
        let paper = lewis_weights(&rows, t);
        for i in 0..rows.len() {
            let ratio = paper[i] / long[i];
            assert!(
                ratio >= 0.5 && ratio <= 2.0,
                "paper T={t} weight {i} ratio {ratio} is outside factor 2"
            );
        }
    }

    #[test]
    fn cone_search_matches_brute_force_net() {
        let dirs = fixed_r3_dirs();
        for l_delta in [5i64, 8, 12] {
            for dir in &dirs {
                let brute = best_net_cosine_brute(dir, l_delta);
                let cone = best_net_cosine_cone(dir, l_delta);
                assert!(
                    (brute - cone).abs() <= 1e-12,
                    "net search mismatch L={l_delta}: brute={brute} cone={cone} dir={dir:?}"
                );
            }
        }
    }

    fn fixed_r3_dirs() -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        for a in -2i32..=2 {
            for b in -2i32..=2 {
                for c in -2i32..=2 {
                    if a == 0 && b == 0 && c == 0 {
                        continue;
                    }
                    // A fixed subset keeps the brute-force comparison small.
                    if (a + 2) + (b + 2) * 2 + (c + 2) * 3 % 3 != 0 && (a * b + c).abs() % 2 != 0
                    {
                        continue;
                    }
                    let v = [a as f64, b as f64, c as f64];
                    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    out.push(vec![v[0] / n, v[1] / n, v[2] / n]);
                }
            }
        }
        assert!(out.len() >= 8);
        out
    }

    fn hand_generators() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![-1.0, 0.5],
            vec![0.3, -0.7],
            vec![2.0, 0.1],
            vec![-0.4, -0.2],
            vec![0.5, 0.8],
            vec![1.2, -1.5],
            vec![-0.9, 0.3],
            vec![0.2, 0.2],
            vec![-1.1, -0.6],
        ]
    }

    #[test]
    fn compression_containment_on_fixed_generators() {
        let original = hand_generators();
        let eta = 0.3;
        let compressed = compress(&original, eta);
        assert!(!compressed.is_empty());
        for k in 0..32 {
            let theta = (k as f64) * std::f64::consts::PI / 16.0;
            let direction = vec![theta.cos(), theta.sin()];
            let s_hat = support(&compressed, &direction);
            let s = support(&original, &direction);
            assert!(
                s_hat <= s * (1.0 + 1e-8),
                "compressed support exceeded original at theta index {k}: {s_hat} > {s}"
            );
            assert!(
                s <= (1.0 + eta) * s_hat * (1.0 + 1e-6),
                "original support exceeded (1+eta) compressed at theta index {k}: {s} > {}",
                (1.0 + eta) * s_hat
            );
        }
    }

    fn expand_root(p: &ProductMixture, q: &ProductMixture) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = p.marginals[0].len();
        let q_size = p.marginals[0][0].len();
        let k1 = p.weights.len();
        let k2 = q.weights.len();
        let k = k1 + k2;
        let mut gamma = Vec::with_capacity(k);
        for &alpha in &p.weights {
            gamma.push(alpha);
        }
        for &beta in &q.weights {
            gamma.push(-beta);
        }
        let mut gens = Vec::new();
        for_each_atom(n, q_size, &mut |atom| {
            let mut g = vec![1.0; k];
            for a in 0..k1 {
                for (i, &symbol) in atom.iter().enumerate() {
                    g[a] *= p.marginals[a][i][symbol];
                }
            }
            for t in 0..k2 {
                for (i, &symbol) in atom.iter().enumerate() {
                    g[k1 + t] *= q.marginals[t][i][symbol];
                }
            }
            gens.push(g);
        });
        (gens, gamma)
    }

    fn root_instance() -> (ProductMixture, ProductMixture) {
        let p = mix(
            vec![0.4, 0.6],
            vec![
                vec![vec![0.2, 0.8], vec![0.5, 0.5], vec![0.9, 0.1]],
                vec![vec![0.7, 0.3], vec![0.4, 0.6], vec![0.2, 0.8]],
            ],
        );
        let q = mix(
            vec![1.0],
            vec![vec![vec![0.5, 0.5], vec![0.3, 0.7], vec![0.6, 0.4]]],
        );
        (p, q)
    }

    #[test]
    fn uncompressed_root_matches_exact_tv() {
        let (p, q) = root_instance();
        let (gens, gamma) = expand_root(&p, &q);
        let from_support = 0.5 * support(&gens, &gamma);
        let exact = exact_tv_product(&p, &q);
        assert!((from_support - exact).abs() <= 1e-12);
        assert!(exact > 0.0);
    }

    #[test]
    fn exact_tv_matches_a_hand_count() {
        let p = mix(vec![1.0], vec![vec![vec![0.2, 0.8]]]);
        let q = mix(vec![1.0], vec![vec![vec![0.6, 0.4]]]);
        let exact = exact_tv_product(&p, &q);
        assert!((exact - 0.4).abs() <= 1e-15);
    }

    fn equal_binary() -> ProductMixture {
        mix(
            vec![1.0],
            vec![vec![vec![0.3, 0.7], vec![0.4, 0.6]]],
        )
    }

    #[test]
    fn fptas_identical_single_component_is_zero() {
        let p = equal_binary();
        let q = equal_binary();
        let eps = 0.5;
        let exact = exact_tv_product(&p, &q);
        let approx = approx_tv_product(&p, &q, eps);
        assert_eq!(exact, 0.0);
        assert!(approx.abs() <= 1e-9);
    }

    fn differing_pair() -> (ProductMixture, ProductMixture) {
        let p = mix(
            vec![1.0],
            vec![vec![vec![0.2, 0.8], vec![0.5, 0.5]]],
        );
        let q = mix(
            vec![1.0],
            vec![vec![vec![0.6, 0.4], vec![0.1, 0.9]]],
        );
        (p, q)
    }

    #[test]
    fn fptas_two_product_distributions() {
        let (p, q) = differing_pair();
        let eps = 0.5;
        let exact = exact_tv_product(&p, &q);
        let approx = approx_tv_product(&p, &q, eps);
        assert!(exact > 1e-6);
        assert!(approx <= exact + 1e-8);
        assert!(exact <= (1.0 + eps) * approx + 1e-8);
        let again = approx_tv_product(&p, &q, eps);
        assert_eq!(approx, again);
    }

    #[test]
    fn fptas_two_components_against_one() {
        let p = mix(
            vec![0.3, 0.7],
            vec![
                vec![vec![0.1, 0.9], vec![0.8, 0.2]],
                vec![vec![0.6, 0.4], vec![0.3, 0.7]],
            ],
        );
        let q = mix(
            vec![1.0],
            vec![vec![vec![0.5, 0.5], vec![0.4, 0.6]]],
        );
        let eps = 0.8;
        let exact = exact_tv_product(&p, &q);
        let approx = approx_tv_product(&p, &q, eps);
        assert!(exact > 1e-6);
        assert!(approx <= exact + 1e-8, "approx {approx} exact {exact}");
        assert!(
            exact <= (1.0 + eps) * approx + 1e-8,
            "exact {exact} bound {} approx {approx}",
            (1.0 + eps) * approx
        );
    }

    #[test]
    fn identical_two_and_two_components_are_zero() {
        let shared = mix(
            vec![0.25, 0.75],
            vec![
                vec![vec![0.2, 0.8], vec![0.4, 0.6]],
                vec![vec![0.7, 0.3], vec![0.1, 0.9]],
            ],
        );
        let approx = approx_tv_product(&shared, &shared, 0.5);
        assert!(
            approx.abs() <= 1e-9,
            "identical mixtures gave approx {approx}"
        );
        assert_eq!(exact_tv_product(&shared, &shared), 0.0);
    }

    #[test]
    #[should_panic]
    fn rejects_weights_that_do_not_sum_to_one() {
        let p = mix(vec![0.5, 0.2], vec![vec![vec![0.5, 0.5]], vec![vec![0.5, 0.5]]]);
        let q = equal_binary();
        let _ = exact_tv_product(&p, &q);
    }

    #[test]
    #[should_panic]
    fn rejects_negative_weight() {
        let p = mix(
            vec![-0.2, 1.2],
            vec![vec![vec![0.5, 0.5]], vec![vec![0.4, 0.6]]],
        );
        let q = mix(vec![1.0], vec![vec![vec![0.5, 0.5]]]);
        let _ = exact_tv_product(&p, &q);
    }

    #[test]
    #[should_panic]
    fn rejects_marginal_that_does_not_sum_to_one() {
        let p = mix(vec![1.0], vec![vec![vec![0.2, 0.2]]]);
        let q = mix(vec![1.0], vec![vec![vec![0.5, 0.5]]]);
        let _ = exact_tv_product(&p, &q);
    }

    #[test]
    #[should_panic]
    fn rejects_eps_outside_unit_interval() {
        let p = equal_binary();
        let _ = approx_tv_product(&p, &p, 0.0);
    }

    #[test]
    #[should_panic]
    fn rejects_eps_equal_to_one() {
        let p = equal_binary();
        let _ = approx_tv_product(&p, &p, 1.0);
    }

    #[test]
    #[should_panic]
    fn rejects_shape_mismatch() {
        let p = equal_binary();
        let q = mix(vec![1.0], vec![vec![vec![0.5, 0.5]]]);
        let _ = exact_tv_product(&p, &q);
    }
}
