use std::collections::BTreeSet;

use semi_ladder_cds::*;

fn set(xs: &[usize]) -> BTreeSet<usize> {
    xs.iter().copied().collect()
}

fn paper_stars(k: usize, m: usize) -> (SetSystem, Vec<Vec<usize>>) {
    // Remark 16 / Section 3.1.1. Universe {t} ∪ {u_i} ∪ {b_ij}.
    // B = {t} ∪ {b_ij}, S_ij = {u_i, b_ij}.
    // Index 0 is t, then u_i, then b_ij in row-major order.
    let mut elements = Vec::new();
    let t = 0usize;
    elements.push(t);
    let mut u = Vec::new();
    for i in 0..k {
        u.push(elements.len());
        elements.push(i + 1);
    }
    let mut b = vec![vec![0usize; m]; k];
    for i in 0..k {
        for j in 0..m {
            b[i][j] = elements.len();
            elements.push(1000 + i * m + j);
        }
    }
    let universe = 1 + k + k * m;
    assert_eq!(elements.len(), universe);
    let mut sets = Vec::new();
    let mut big = BTreeSet::new();
    big.insert(0);
    for i in 0..k {
        for j in 0..m {
            big.insert(b[i][j]);
        }
    }
    sets.push(big);
    let mut index = vec![vec![0usize; m]; k];
    for i in 0..k {
        for j in 0..m {
            index[i][j] = sets.len();
            sets.push(set(&[u[i], b[i][j]]));
        }
    }
    let mut covers = Vec::new();
    let mut choice = vec![0usize; k];
    fn rec(
        i: usize,
        k: usize,
        m: usize,
        index: &[Vec<usize>],
        choice: &mut [usize],
        covers: &mut Vec<Vec<usize>>,
    ) {
        if i == k {
            let mut c = vec![0];
            for r in 0..k {
                c.push(index[r][choice[r]]);
            }
            c.sort_unstable();
            covers.push(c);
            return;
        }
        for j in 0..m {
            choice[i] = j;
            rec(i + 1, k, m, index, choice, covers);
        }
    }
    rec(0, k, m, &index, &mut choice, &mut covers);
    (SetSystem::new(universe, sets), covers)
}

fn spec_stars(k: usize, m: usize) -> (SetSystem, Vec<Vec<usize>>) {
    // Spec wording: c together with e_ij, B = all e_ij, S_ij = {c} ∪ {e_ij}.
    let universe = 1 + k * m;
    let mut sets = Vec::new();
    let mut big = BTreeSet::new();
    for i in 0..k {
        for j in 0..m {
            big.insert(1 + i * m + j);
        }
    }
    sets.push(big);
    let mut covers = Vec::new();
    let mut choice = vec![0usize; k];
    let mut index = vec![vec![0usize; m]; k];
    for i in 0..k {
        for j in 0..m {
            index[i][j] = sets.len();
            sets.push(set(&[0, 1 + i * m + j]));
        }
    }
    fn rec(
        i: usize,
        k: usize,
        m: usize,
        index: &[Vec<usize>],
        choice: &mut [usize],
        covers: &mut Vec<Vec<usize>>,
    ) {
        if i == k {
            let mut c = vec![0];
            for r in 0..k {
                c.push(index[r][choice[r]]);
            }
            c.sort_unstable();
            covers.push(c);
            return;
        }
        for j in 0..m {
            choice[i] = j;
            rec(i + 1, k, m, index, choice, covers);
        }
    }
    rec(0, k, m, &index, &mut choice, &mut covers);
    (SetSystem::new(universe, sets), covers)
}

fn brute_minimal_covers(system: &SetSystem, max_k: usize) -> Vec<Vec<usize>> {
    let m = system.sets.len();
    assert!(m <= 20, "brute force cap");
    let mut out = Vec::new();
    for mask in 1usize..(1usize << m) {
        let bits = mask.count_ones() as usize;
        if bits > max_k {
            continue;
        }
        let idxs: Vec<usize> = (0..m).filter(|i| mask & (1usize << i) != 0).collect();
        if !covers_universe(system, &idxs) {
            continue;
        }
        let minimal = idxs.iter().all(|&drop_i| {
            let sub: Vec<usize> = idxs.iter().copied().filter(|&j| j != drop_i).collect();
            !covers_universe(system, &sub)
        });
        if minimal {
            out.push(idxs);
        }
    }
    out
}

fn brute_connected_set_cover(system: &SetSystem, k: usize) -> Option<Vec<usize>> {
    let m = system.sets.len();
    let ig = {
        let mut g = Graph::new(m);
        for i in 0..m {
            for j in (i + 1)..m {
                if system.sets[i].intersection(&system.sets[j]).next().is_some() {
                    g.add_edge(i, j);
                }
            }
        }
        g
    };
    let mut best = None;
    for mask in 0usize..(1usize << m) {
        let bits = mask.count_ones() as usize;
        if bits > k {
            continue;
        }
        let idxs: Vec<usize> = (0..m).filter(|i| mask & (1usize << i) != 0).collect();
        if system.universe == 0 && bits == 0 {
            return Some(Vec::new());
        }
        if bits == 0 {
            continue;
        }
        if covers_universe(system, &idxs) && set_is_connected(&ig, &idxs) {
            if best
                .as_ref()
                .map(|b: &Vec<usize>| idxs.len() < b.len() || (idxs.len() == b.len() && idxs < *b))
                .unwrap_or(true)
            {
                best = Some(idxs);
            }
        }
    }
    best
}

fn assert_representation_sound(system: &SetSystem, k: usize, d: usize) {
    let rep = compact_representation(system, k, d);
    let minimal = brute_minimal_covers(system, k);
    for cover in &minimal {
        let hit = rep.iter().any(|t| encapsulates_cover(t, cover));
        assert!(
            hit,
            "minimal cover {cover:?} not encapsulated by {rep:?}"
        );
    }
    for tuple in &rep {
        let mut seen = BTreeSet::new();
        for fam in tuple {
            for &idx in fam {
                assert!(seen.insert(idx), "families not disjoint: {tuple:?}");
            }
        }
        if tuple.iter().any(|f| f.is_empty()) {
            continue;
        }
        let product = tuple_product_size(tuple);
        if product <= 4096 {
            for seq in tuple_product(tuple) {
                assert!(
                    covers_universe(system, &seq),
                    "sequence {seq:?} from {tuple:?} does not cover"
                );
            }
        }
    }
}

fn tuple_product_size(tuple: &Tuple) -> usize {
    tuple.iter().fold(1usize, |acc, f| acc.saturating_mul(f.len()))
}

fn tuple_product(tuple: &Tuple) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for fam in tuple {
        let mut next = Vec::new();
        for prefix in &out {
            for &s in fam {
                let mut row = prefix.clone();
                row.push(s);
                next.push(row);
            }
        }
        out = next;
    }
    out
}

#[test]
fn semi_ladder_witness_and_index() {
    let system = SetSystem::from_slices(3, &[vec![0, 1], vec![1, 2], vec![0, 1, 2]]);
    assert!(is_d_semi_ladder_free(&system, 3));
    assert!(!is_d_semi_ladder_free(&system, 1));
    let w = semi_ladder_witness(&system, 2).expect("2-semi-ladder");
    assert_eq!(w.set_indices.len(), 2);
    assert_eq!(w.elements.len(), 2);
    let s0 = &system.sets[w.set_indices[0]];
    let s1 = &system.sets[w.set_indices[1]];
    assert!(!s0.contains(&w.elements[0]));
    assert!(s1.contains(&w.elements[0]));
    assert!(!s1.contains(&w.elements[1]));

    let path = Graph::from_edges(3, [(0, 1), (1, 2)]);
    let idx = semi_ladder_index(&path).unwrap();
    assert!(idx >= 1);
    assert!(is_semi_ladder_free(&path, idx + 1));
    assert!(is_semi_ladder_free_with_witness(&path, idx).is_some() || idx == 0);
    if idx > 0 {
        assert!(is_semi_ladder_free_with_witness(&path, idx).is_some());
    }
}

#[test]
fn stars_representation_is_a_single_tuple() {
    let (system, covers) = paper_stars(2, 3);
    assert_eq!(covers.len(), 9);
    let d = set_semi_ladder_index(&system) + 1;
    assert!(is_d_semi_ladder_free(&system, d));
    let rep = compact_representation(&system, 3, d);
    assert!(
        rep.len() <= 1,
        "expected at most one tuple, got {rep:?}"
    );
    assert_eq!(rep.len(), 1);
    for cover in &covers {
        assert!(
            encapsulates_cover(&rep[0], cover),
            "cover {cover:?} not in {:?}",
            rep[0]
        );
    }
    assert_representation_sound(&system, 3, d);
    let minimal = brute_minimal_covers(&system, 3);
    assert_eq!(minimal.len(), covers.len());
}

#[test]
fn spec_stars_covers_are_encapsulated() {
    // Spec construction with k = 1: the m covers that use B are minimal and
    // have size k+1. (For k > 1 that construction's B-using m^k lists are not
    // minimal, because one S-set already covers the special element. Remark 16's
    // family, tested above, is the construction in which those lists are minimal.)
    let (system, covers) = spec_stars(1, 4);
    assert_eq!(covers.len(), 4);
    let d = set_semi_ladder_index(&system) + 1;
    let rep = compact_representation(&system, 2, d);
    assert!(rep.len() <= 1, "expected at most one tuple, got {rep:?}");
    for cover in &covers {
        assert!(
            rep.iter().any(|t| encapsulates_cover(t, cover)),
            "missing {cover:?} in {rep:?}"
        );
    }
    assert_representation_sound(&system, 2, d);
}

#[test]
fn branching_is_forced_on_four_elements() {
    // No single set covers, so the empty seed is not a latent cover and Solve must branch.
    let system = SetSystem::from_slices(
        4,
        &[vec![0, 1], vec![2, 3], vec![0, 2], vec![1, 3]],
    );
    assert!(semi_ladder_witness(&system, 1).is_some());
    let d = set_semi_ladder_index(&system) + 1;
    let rep = compact_representation(&system, 2, d);
    assert!(!rep.is_empty(), "branching produced no tuple");
    let minimal = brute_minimal_covers(&system, 2);
    assert!(minimal.iter().any(|c| c == &vec![0, 1]));
    assert!(minimal.iter().any(|c| c == &vec![2, 3]));
    assert_representation_sound(&system, 2, d);
    // The base all-empty seed does not cover, so some family is a proper subset of F.
    assert!(rep.iter().any(|t| t.iter().any(|f| f.len() < system.sets.len())));
}

#[test]
fn representation_bound_and_random_completeness() {
    let samples = [
        SetSystem::from_slices(3, &[vec![0], vec![1], vec![2], vec![0, 1], vec![1, 2]]),
        SetSystem::from_slices(4, &[vec![0, 1], vec![1, 2], vec![2, 3], vec![0, 3], vec![0, 1, 2]]),
        SetSystem::from_slices(2, &[vec![0], vec![1], vec![0, 1]]),
        SetSystem::from_slices(5, &[vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 0]]),
    ];
    for system in &samples {
        let d = set_semi_ladder_index(system) + 1;
        let k = 3;
        let rep = compact_representation(system, k, d);
        let bound = k.saturating_pow((k * d + 1) as u32);
        assert!(
            rep.len() <= bound,
            "|D|={} exceeds k^(kd+1)={bound} for d={d}",
            rep.len()
        );
        assert_representation_sound(system, k, d);
    }
}

#[test]
fn group_steiner_tree_path_and_star() {
    let star = Graph::from_edges(3, [(0, 1), (1, 2)]);
    let t = group_steiner_tree(&star, &[vec![0], vec![2]]).unwrap();
    assert_eq!(t, vec![0, 1, 2]);

    let path = Graph::from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]);
    let t = group_steiner_tree(&path, &[vec![0], vec![4]]).unwrap();
    assert!(t.contains(&2), "center missing from {t:?}");
    assert_eq!(t, vec![0, 1, 2, 3, 4]);

    let edge = Graph::from_edges(2, [(0, 1)]);
    let t = group_steiner_tree(&edge, &[vec![0], vec![1]]).unwrap();
    assert_eq!(t, vec![0, 1]);

    assert!(group_steiner_tree(&path, &[vec![0], vec![]]).is_none());
    let only = group_steiner_tree(&path, &[vec![3, 1]]).unwrap();
    assert_eq!(only, vec![1]);
}

#[test]
fn connected_set_cover_matches_brute_force() {
    // Exhaustive on n ≤ 3, |F| ≤ 3, and a sample up to n = 6, |F| = 6.
    for n in 1..=3 {
        let subsets = all_subsets(n);
        let max_f = 3;
        for fcount in 0..=max_f {
            let mut choice = vec![0usize; fcount];
            enumerate_families(n, &subsets, &mut choice, &mut |system| {
                for k in 1..=3 {
                    let d = n + 1;
                    let got = connected_set_cover(system, k, d);
                    let expect = brute_connected_set_cover(system, k);
                    match (&got, &expect) {
                        (None, None) => {}
                        (Some(sol), Some(_)) => {
                            assert!(sol.len() <= k);
                            assert!(covers_universe(system, sol));
                            let ig = intersection_graph_of(system);
                            assert!(set_is_connected(&ig, sol));
                        }
                        (got, expect) => panic!("n={n} k={k} system={system:?} got={got:?} expect={expect:?}"),
                    }
                }
            });
        }
    }
    let mut seed = 0x1234_5678u64;
    for n in [4usize, 5, 6] {
        for _ in 0..12 {
            let fcount = 1 + (lcg(&mut seed) as usize % 6);
            let mut sets = Vec::new();
            for _ in 0..fcount {
                let mut s = Vec::new();
                for x in 0..n {
                    seed = lcg(&mut seed);
                    if seed % 3 != 0 {
                        s.push(x);
                    }
                }
                sets.push(s);
            }
            let system = SetSystem::from_slices(n, &sets);
            for k in 1..=3 {
                let got = connected_set_cover(&system, k, n + 1);
                let expect = brute_connected_set_cover(&system, k);
                match (&got, &expect) {
                    (None, None) => {}
                    (Some(sol), Some(_)) => {
                        assert!(sol.len() <= k && covers_universe(&system, sol));
                    }
                    other => panic!("mismatch {other:?} on {system:?}"),
                }
            }
        }
    }
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn all_subsets(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for mask in 0..(1usize << n) {
        out.push((0..n).filter(|i| mask & (1usize << i) != 0).collect());
    }
    out
}

fn enumerate_families(n: usize, subsets: &[Vec<usize>], choice: &mut [usize], visit: &mut dyn FnMut(&SetSystem)) {
    fn rec(
        n: usize,
        subsets: &[Vec<usize>],
        choice: &mut [usize],
        i: usize,
        visit: &mut dyn FnMut(&SetSystem),
    ) {
        if i == choice.len() {
            let sets: Vec<Vec<usize>> = choice.iter().map(|&c| subsets[c].clone()).collect();
            visit(&SetSystem::from_slices(n, &sets));
            return;
        }
        let start = if i == 0 { 0 } else { choice[i - 1] };
        for c in start..subsets.len() {
            choice[i] = c;
            rec(n, subsets, choice, i + 1, visit);
        }
    }
    rec(n, subsets, choice, 0, visit);
}

fn intersection_graph_of(system: &SetSystem) -> Graph {
    let m = system.sets.len();
    let mut g = Graph::new(m);
    for i in 0..m {
        for j in (i + 1)..m {
            if system.sets[i].intersection(&system.sets[j]).next().is_some() {
                g.add_edge(i, j);
            }
        }
    }
    g
}

#[test]
fn grouping_shrinks_and_preserves_freeness() {
    let g = Graph::from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]);
    let d = semi_ladder_index(&g).unwrap() + 1;
    assert!(is_semi_ladder_free(&g, d));
    let grouped = group(&g, &[1, 3]);
    assert!(grouped.n() < g.n());
    assert_eq!(grouped.n(), g.n() - 1);
    let d_after = semi_ladder_index(&grouped).unwrap();
    assert!(
        d_after < d,
        "grouping raised the semi-ladder index from {} to {d_after}",
        d - 1
    );
    assert!(is_semi_ladder_free(&grouped, d));
    let core = find_core(&g, 2, 2);
    assert!(!core.is_empty() || g.n() == 0);
    let again = find_core(&grouped, 1, 1);
    assert!(again.len() <= grouped.n());
}

#[test]
fn lossy_compression_lifts_on_a_small_graph() {
    let g = Graph::from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]);
    let d = semi_ladder_index(&g).unwrap() + 1;
    let k = 2;
    let inst = lossy_cds_compression_instance(&g, k, d, 1.0);
    let out_index = semi_ladder_index(&inst.graph).unwrap();
    assert!(
        out_index <= d + 1,
        "output index {out_index} exceeds d+2 freeness threshold (need index ≤ d+1 = {})",
        d + 1
    );
    assert!(is_semi_ladder_free(&inst.graph, d + 2));
    let cds = minimum_connected_dominating_set(&inst.graph).expect("compressed graph has a CDS");
    let lifted = lift_connected_dominating_set(&inst, &cds, &g).expect("lift");
    assert!(
        is_connected_dominating_set(&g, &lifted),
        "lifted {lifted:?} is not a CDS of the input"
    );
    assert!(lifted.len() <= cds.len());
}

#[test]
fn domination_predicates() {
    let p3 = Graph::from_edges(3, [(0, 1), (1, 2)]);
    assert!(!is_complete(&p3));
    assert!(is_connected(&p3));
    assert_eq!(count_components(&p3), 1);
    assert_eq!(dominating_set_size(&p3, &[1]), Some(1));
    assert_eq!(connected_dominating_set_size(&p3, &[1]), Some(1));
    assert!(is_minimum_dominating_set(&p3, &[1]));
    assert!(is_minimum_connected_dominating_set(&p3, &[1]));
    assert!(!is_minimum_connected_dominating_set(&p3, &[0, 1, 2]));
    assert_eq!(connected_domination_number(&p3), Some(1));
    assert_eq!(domination_number(&p3), Some(1));

    let k3 = Graph::from_edges(3, [(0, 1), (1, 2), (0, 2)]);
    assert!(is_complete(&k3));
    assert_eq!(deletion_core(&k3, 1), Some(vec![0]));
    assert!(deletion_core(&p3, 1).is_none() || is_semi_ladder_free(&p3, 1));

    let disconnected = Graph::from_edges(3, [(0, 1)]);
    assert_eq!(count_components(&disconnected), 2);
    assert!(connected_domination_number(&disconnected).is_none());
    assert!(!is_connected(&disconnected));
}
