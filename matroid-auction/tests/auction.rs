use matroid_auction::{
    approximate, auction, ceil_lambda, min_pi_circuit_element, IndependenceOracle, PartitionMatroid,
    TransversalMatroid, UniformMatroid,
};

fn bucket(big_w: f64, eps: f64, k: i64) -> f64 {
    big_w * (1.0 + eps).powf(-(k as f64))
}

fn assert_ceil_props(u: f64, eps: f64, big_w: f64) {
    let got = ceil_lambda(u, eps, big_w);
    if u <= 0.0 {
        assert_eq!(got, 0.0, "ceil of nonpositive u");
        return;
    }
    assert!(
        got >= u,
        "ceil {got} is below u {u} (eps={eps}, W={big_w})"
    );
    let mut matched = None;
    for k in 0..20_000 {
        let lam = bucket(big_w, eps, k);
        let rel = (got - lam).abs() / lam;
        if rel <= 1e-9 {
            matched = Some(k);
            break;
        }
        if lam < got * 1e-6 && lam < u * 1e-6 {
            break;
        }
    }
    let k = matched.unwrap_or_else(|| panic!("ceil {got} not in Lambda for u={u}"));
    let prev = bucket(big_w, eps, k + 1);
    assert!(
        prev < u,
        "previous bucket {prev} is not < u {u} (got {got}, k={k})"
    );
}

#[test]
fn ceil_matches_lambda() {
    let cases = [(0.1, 8.0), (0.05, 2.0), (0.25, 20.0), (0.1, 1.0)];
    for (eps, big_w) in cases {
        assert_ceil_props(0.0, eps, big_w);
        assert_ceil_props(-3.0, eps, big_w);
        assert_ceil_props(big_w, eps, big_w);
        for k in 0..30 {
            let b = bucket(big_w, eps, k);
            assert_ceil_props(b, eps, big_w);
            let below = b * (1.0 - 1e-9);
            if below > 0.0 {
                assert_ceil_props(below, eps, big_w);
                let got_below = ceil_lambda(below, eps, big_w);
                let rel = (got_below - b).abs() / b;
                assert!(
                    rel <= 1e-9,
                    "just below bucket k={k}: got {got_below}, bucket {b}"
                );
            }
            if k > 0 {
                let above = b * (1.0 + 1e-9);
                if above <= big_w {
                    assert_ceil_props(above, eps, big_w);
                    let larger = bucket(big_w, eps, k - 1);
                    let got_above = ceil_lambda(above, eps, big_w);
                    let rel = (got_above - larger).abs() / larger;
                    assert!(
                        rel <= 1e-9,
                        "just above bucket k={k}: got {got_above}, larger {larger}"
                    );
                }
            }
            if k + 1 < 30 {
                let mid = (b * bucket(big_w, eps, k + 1)).sqrt();
                if mid > 0.0 && mid < b {
                    assert_ceil_props(mid, eps, big_w);
                }
            }
        }
    }
}

fn naive_min_pi<O: IndependenceOracle>(
    oracle: &O,
    x: &[usize],
    e: usize,
    pi: &[f64],
) -> Option<usize> {
    let mut with: Vec<usize> = x.iter().copied().filter(|&f| f != e).collect();
    with.push(e);
    if oracle.independent(&with) {
        return None;
    }
    let mut best: Option<usize> = None;
    for &f in x {
        if f == e {
            continue;
        }
        let set: Vec<usize> = x
            .iter()
            .copied()
            .filter(|&g| g != f && g != e)
            .chain(std::iter::once(e))
            .collect();
        if oracle.independent(&set) {
            best = Some(match best {
                None => f,
                Some(b) => {
                    if pi[f] < pi[b] || (pi[f] == pi[b] && f < b) {
                        f
                    } else {
                        b
                    }
                }
            });
        }
    }
    best
}

fn assert_circuit_match<O: IndependenceOracle>(oracle: &O, x: &[usize], e: usize, pi: &[f64]) {
    let naive = naive_min_pi(oracle, x, e, pi);
    let fast = min_pi_circuit_element(oracle, x, e, pi);
    assert_eq!(naive, fast, "circuit mismatch x={x:?} e={e} pi={pi:?}");
}

#[test]
fn binary_search_circuit_matches_naive_scan() {
    let n = 6;
    let part = PartitionMatroid::new(
        n,
        &[
            (vec![0, 1, 2], 2),
            (vec![3], 1),
            (vec![4, 5], 1),
        ],
    );
    // Circuit {0, 1, 2}. Duplicate prices, smaller index wins.
    let pi = vec![1.0, 1.0, 0.0, 0.01, 4.0, 4.0];
    assert_circuit_match(&part, &[0, 1], 2, &pi);
    assert_eq!(min_pi_circuit_element(&part, &[0, 1], 2, &pi), Some(0));

    // Strictly smaller price on the other circuit element.
    let pi = vec![2.5, 0.4, 9.0, 0.01, 1.0, 1.0];
    assert_circuit_match(&part, &[0, 1, 3], 2, &pi);
    assert_eq!(min_pi_circuit_element(&part, &[0, 1, 3], 2, &pi), Some(1));

    // Low price outside the circuit must not win. 3 is free of group {0,1,2}.
    let pi = vec![5.0, 5.0, 7.0, 0.1, 3.0, 3.0];
    assert_circuit_match(&part, &[0, 1, 3], 2, &pi);
    assert_eq!(min_pi_circuit_element(&part, &[0, 1, 3], 2, &pi), Some(0));

    // Unsorted X, duplicate prices on both circuit elements. The smaller index
    // is not first in the slice, and a cheaper element outside the circuit
    // must not win.
    let pi = vec![2.0, 2.0, 8.0, 0.05, 1.0, 8.0];
    assert_circuit_match(&part, &[1, 0, 3], 2, &pi);
    assert_eq!(min_pi_circuit_element(&part, &[1, 0, 3], 2, &pi), Some(0));

    // Independent: no circuit.
    assert_circuit_match(&part, &[0, 3], 4, &pi);
    assert_eq!(min_pi_circuit_element(&part, &[0, 3], 4, &pi), None);

    // Capacity-1 circuit with a duplicate-price decoy in another group.
    let part2 = PartitionMatroid::new(4, &[(vec![0, 1], 1), (vec![2, 3], 1)]);
    let pi = vec![1.5, 1.5, 0.2, 8.0];
    assert_circuit_match(&part2, &[0, 2], 1, &pi);
    assert_eq!(min_pi_circuit_element(&part2, &[0, 2], 1, &pi), Some(0));

    // Transversal: rights 0, 1 cover a 3-circuit; right 2 isolates element 3.
    let trans = TransversalMatroid::new(
        3,
        vec![
            vec![0, 1],
            vec![0, 1],
            vec![0, 1],
            vec![2],
            vec![0],
            vec![1, 2],
        ],
    );
    let pi = vec![5.0, 5.0, 1.0, 0.1, 4.0, 4.0];
    assert_circuit_match(&trans, &[0, 1, 3], 2, &pi);
    assert_eq!(min_pi_circuit_element(&trans, &[0, 1, 3], 2, &pi), Some(0));

    let pi = vec![3.0, 1.25, 9.0, 0.05, 2.0, 2.0];
    assert_circuit_match(&trans, &[0, 1, 3], 2, &pi);
    assert_eq!(min_pi_circuit_element(&trans, &[0, 1, 3], 2, &pi), Some(1));

    // Duplicate prices, unsorted, both circuit members tied.
    let pi = vec![2.0, 2.0, 6.0, 0.3, 1.0, 7.0];
    assert_circuit_match(&trans, &[1, 3, 0], 2, &pi);
    assert_eq!(min_pi_circuit_element(&trans, &[1, 3, 0], 2, &pi), Some(0));

    // Independent transversal set.
    assert_circuit_match(&trans, &[3], 0, &pi);
    assert_eq!(min_pi_circuit_element(&trans, &[3], 0, &pi), None);

    // A second transversal whose circuit is not the whole ground set.
    let trans2 = TransversalMatroid::new(
        3,
        vec![vec![0], vec![0, 1], vec![1], vec![2], vec![0, 2]],
    );
    let pi = vec![1.0, 1.0, 4.0, 0.2, 3.0];
    // X = {0, 1, 3} is matchable (0->0, 1->1, 3->2). e=2 shares rights with {0,1}.
    assert_circuit_match(&trans2, &[0, 1, 3], 2, &pi);
}

fn top_weight(weights: &[f64], r: usize) -> f64 {
    let mut w = weights.to_vec();
    w.sort_by(|a, b| b.partial_cmp(a).unwrap());
    w.iter().take(r).sum()
}

fn assert_uniform_pair(weights: &[f64], k1: usize, k2: usize) {
    let opt = top_weight(weights, k1.min(k2));
    let m1 = UniformMatroid::new(k1);
    let m2 = UniformMatroid::new(k2);
    let sol = auction(weights, &m1, &m2, 0.1).expect("auction");
    assert!(
        sol.weight + 1e-8 >= 0.5 * opt,
        "auction weight {} < 0.5 * OPT {} on {:?}",
        sol.weight,
        opt,
        weights
    );
    assert!(m1.independent(&sol.set));
    assert!(m2.independent(&sol.set));
    let again = approximate(weights, &m1, &m2, 0.5).expect("approximate");
    assert!(
        again.weight + 1e-8 >= 0.5 * opt,
        "approximate weight {} < 0.5 * OPT {} on {:?}",
        again.weight,
        opt,
        weights
    );
    assert!(m1.independent(&again.set));
    assert!(m2.independent(&again.set));
}

#[test]
fn uniform_uniform_half_opt() {
    let w = vec![1.0, 2.0, 3.0];
    assert!(auction(&w, &UniformMatroid::new(1), &UniformMatroid::new(1), 0.1 + 1e-9).is_err());
    assert!(auction(&w, &UniformMatroid::new(1), &UniformMatroid::new(1), 0.0).is_err());
    assert!(auction(&w, &UniformMatroid::new(1), &UniformMatroid::new(1), -0.05).is_err());
    assert!(approximate(&w, &UniformMatroid::new(1), &UniformMatroid::new(1), 0.5 + 1e-9).is_err());
    assert!(approximate(&w, &UniformMatroid::new(1), &UniformMatroid::new(1), 0.0).is_err());

    assert_uniform_pair(&[1.0, 2.0, 3.0, 4.0, 5.0], 2, 2);
    assert_uniform_pair(&[5.0, 1.0, 4.0, 1.0, 3.0], 3, 2);
    assert_uniform_pair(&[1e-6, 1e-6, 1e6, 2e6, 1e-3], 2, 2);
    assert_uniform_pair(&[100.0, 1.0, 1.0, 1.0, 100.0], 1, 2);
    assert_uniform_pair(&[1.0, 1000.0, 1.0, 1000.0, 1.0, 1000.0], 3, 3);
    assert_uniform_pair(&[1e-9, 1e-3, 1.0, 1e9, 1e-6], 1, 3);
    assert_uniform_pair(&[7.0], 1, 1);
    assert_uniform_pair(&[4.0, 4.0, 4.0, 9.0], 2, 4);
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        assert!(lo < hi);
        lo + (self.next() as usize % (hi - lo))
    }
}

fn brute_opt<M1: IndependenceOracle, M2: IndependenceOracle>(
    weights: &[f64],
    m1: &M1,
    m2: &M2,
) -> f64 {
    let n = weights.len();
    assert!(n <= 12);
    let mut best = 0.0;
    let total = 1u32 << n;
    for mask in 0..total {
        let set: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        if m1.independent(&set) && m2.independent(&set) {
            let w: f64 = set.iter().map(|&e| weights[e]).sum();
            if w > best {
                best = w;
            }
        }
    }
    best
}

fn greedy_common<M1: IndependenceOracle, M2: IndependenceOracle>(
    weights: &[f64],
    m1: &M1,
    m2: &M2,
) -> f64 {
    let n = weights.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut set = Vec::new();
    for e in order {
        set.push(e);
        if !(m1.independent(&set) && m2.independent(&set)) {
            set.pop();
        }
    }
    set.iter().map(|&e| weights[e]).sum()
}

fn assert_auction_bar<M1: IndependenceOracle, M2: IndependenceOracle>(
    weights: &[f64],
    m1: &M1,
    m2: &M2,
    opt: f64,
) {
    let sol = auction(weights, m1, m2, 0.1).expect("auction");
    assert!(
        sol.weight + 1e-8 >= 0.5 * opt,
        "auction {} < 0.5 * OPT {} weights={:?}",
        sol.weight,
        opt,
        weights
    );
    assert!(
        m1.independent(&sol.set),
        "not M1-independent: {:?}",
        sol.set
    );
    assert!(
        m2.independent(&sol.set),
        "not M2-independent: {:?}",
        sol.set
    );
    let recomputed: f64 = sol.set.iter().map(|&e| weights[e]).sum();
    assert!((recomputed - sol.weight).abs() <= 1e-8);
    for &e in &sol.set {
        let lhs = sol.pi[e] + sol.kappa[e];
        let rhs = (1.0 + 2.0 * 0.1) * weights[e];
        assert!(
            (lhs - rhs).abs() <= 1e-8,
            "price-utility identity failed e={e}: {lhs} vs {rhs}"
        );
    }
}

fn random_partition(rng: &mut Lcg, n: usize) -> PartitionMatroid {
    let g = rng.gen_range(2, n.min(5) + 1);
    let mut groups = vec![Vec::new(); g];
    for e in 0..n {
        groups[rng.gen_range(0, g)].push(e);
    }
    let spec: Vec<(Vec<usize>, usize)> = groups
        .into_iter()
        .filter(|grp| !grp.is_empty())
        .map(|grp| {
            let cap = rng.gen_range(1, grp.len() + 1);
            (grp, cap)
        })
        .collect();
    PartitionMatroid::new(n, &spec)
}

fn random_weights(rng: &mut Lcg, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let base = rng.gen_range(1, 200) as f64;
            let scale = [1.0, 10.0, 0.5][rng.gen_range(0, 3)];
            base * scale / 10.0
        })
        .collect()
}

fn random_transversal(rng: &mut Lcg, n: usize) -> TransversalMatroid {
    let r = rng.gen_range(2, (n / 2).max(3) + 1);
    let mut adj = vec![Vec::new(); n];
    for e in 0..n {
        let deg = rng.gen_range(1, r + 1);
        let mut used = vec![false; r];
        for _ in 0..deg {
            let v = rng.gen_range(0, r);
            if !used[v] {
                used[v] = true;
                adj[e].push(v);
            }
        }
        if adj[e].is_empty() {
            adj[e].push(rng.gen_range(0, r));
        }
    }
    TransversalMatroid::new(r, adj)
}

#[test]
fn brute_force_instances_clear_half_opt() {
    let mut rng = Lcg(0xC0FFEE_u64);
    for i in 0..8 {
        let n = 5 + (i % 8);
        // n is 5..=12
        let n = n.min(12);
        let weights = random_weights(&mut rng, n);
        let m1 = random_partition(&mut rng, n);
        let m2 = random_partition(&mut rng, n);
        let opt = brute_opt(&weights, &m1, &m2);
        assert_auction_bar(&weights, &m1, &m2, opt);
    }

    for i in 0..2 {
        let n = 6 + i * 3;
        let weights = random_weights(&mut rng, n);
        let m1 = random_transversal(&mut rng, n);
        let m2 = random_transversal(&mut rng, n);
        let opt = brute_opt(&weights, &m1, &m2);
        assert_auction_bar(&weights, &m1, &m2, opt);
    }

    // Weight-greedy common independent set is not optimal.
    // M1 forbids {0,1}; M2 forbids {1,2}. Greedy takes 1 and stops at weight 3.
    // OPT is {0,2} at weight 4.
    let weights = vec![2.0, 3.0, 2.0];
    let m1 = PartitionMatroid::new(3, &[(vec![0, 1], 1), (vec![2], 1)]);
    let m2 = PartitionMatroid::new(3, &[(vec![0], 1), (vec![1, 2], 1)]);
    let opt = brute_opt(&weights, &m1, &m2);
    let greedy = greedy_common(&weights, &m1, &m2);
    assert!(
        greedy + 1e-8 < opt,
        "expected greedy {greedy} < OPT {opt}"
    );
    assert!((opt - 4.0).abs() <= 1e-8);
    assert_auction_bar(&weights, &m1, &m2, opt);
}

#[test]
fn query_count_is_logarithmic() {
    let n = 10usize;
    let eps = 0.1;
    let weights: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let groups_a: Vec<(Vec<usize>, usize)> = (0..5).map(|g| (vec![2 * g, 2 * g + 1], 1)).collect();
    let groups_b: Vec<(Vec<usize>, usize)> = vec![
        (vec![0, 2, 4, 6, 8], 2),
        (vec![1, 3, 5, 7, 9], 2),
    ];
    let m1 = PartitionMatroid::new(n, &groups_a);
    let m2 = PartitionMatroid::new(n, &groups_b);
    m1.reset_queries();
    m2.reset_queries();
    let sol = auction(&weights, &m1, &m2, eps).expect("auction");
    let queries = m1.queries() + m2.queries();
    let t = (10.0 * eps.powi(-2) * (2.0 * n as f64).ln()).ceil();
    let bound = 8.0 * (n as f64) * t * ((n as f64).log2() + 2.0);
    println!("oracle queries: {queries}");
    assert!(queries > 0, "algorithm did not call the oracles");
    assert!(
        (queries as f64) <= bound,
        "queries {queries} exceed bound {bound} (T={t})"
    );
    assert!(m1.independent(&sol.set));
    assert!(m2.independent(&sol.set));
}

#[test]
fn two_calls_same_output() {
    let weights = vec![2.5, 8.0, 1.0, 6.5, 3.0, 9.0, 4.0];
    let m1 = PartitionMatroid::new(
        7,
        &[(vec![0, 1, 2], 2), (vec![3, 4], 1), (vec![5, 6], 1)],
    );
    let m2 = TransversalMatroid::new(
        4,
        vec![
            vec![0, 1],
            vec![1],
            vec![0, 2],
            vec![2, 3],
            vec![1, 3],
            vec![0],
            vec![2, 3],
        ],
    );
    let a = auction(&weights, &m1, &m2, 0.1).expect("first");
    let b = auction(&weights, &m1, &m2, 0.1).expect("second");
    assert_eq!(a, b);

    let u1 = UniformMatroid::new(3);
    let u2 = UniformMatroid::new(4);
    let c = approximate(&weights, &u1, &u2, 0.5).expect("approx first");
    let d = approximate(&weights, &u1, &u2, 0.5).expect("approx second");
    assert_eq!(c, d);
}
