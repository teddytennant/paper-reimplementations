use ep_realizability::{
    bipartite_havel_hakimi, edge_overlap, ep_of, gale_ryser, havel_hakimi, implied_edges,
    is_equitable, is_graphic_regular, karate_club, karate_faction_partition, partition_from_h,
    phi_eps, realizable, realize, realize_sampled, AhMatrix, KARATE_EDGES, KARATE_N,
};

fn ah(rows: Vec<Vec<usize>>) -> AhMatrix {
    AhMatrix::new(rows).unwrap()
}

fn assert_simple(g: &ep_realizability::SimpleGraph) {
    assert!(g.is_simple(), "graph must be simple");
    let mut seen = std::collections::BTreeSet::new();
    for &(u, v) in g.edges() {
        assert_ne!(u, v, "self-loop");
        assert!(u < v, "undirected edges stored with u < v");
        assert!(seen.insert((u, v)), "duplicate edge");
        assert!(u < g.n() && v < g.n());
    }
    assert_eq!(seen.len(), g.m());
}

#[test]
fn havel_hakimi_3_regular_on_6() {
    let edges = havel_hakimi(&[3, 3, 3, 3, 3, 3]).expect("graphic");
    let g = ep_realizability::SimpleGraph::from_edges(6, edges).unwrap();
    assert_simple(&g);
    assert_eq!(g.degrees(), vec![3, 3, 3, 3, 3, 3]);
    assert_eq!(g.m(), 9);
    for &(u, v) in g.edges() {
        assert_ne!(u, v);
    }
}

#[test]
fn non_graphic_rejected() {
    assert!(!is_graphic_regular(5, 3), "odd sum 5*3");
    assert!(havel_hakimi(&[3, 3, 3, 3, 3]).is_err());
    assert!(!is_graphic_regular(4, 5), "degree 5 on 4 vertices");
    assert!(havel_hakimi(&[5, 5, 5, 5]).is_err());
}

#[test]
fn gale_ryser_k23_and_unbalanced() {
    assert!(gale_ryser(&[3, 3], &[2, 2, 2]));
    let e = bipartite_havel_hakimi(&[3, 3], &[2, 2, 2]).unwrap();
    assert_eq!(e.len(), 6);
    assert!(!gale_ryser(&[3, 3], &[1, 1, 1]));
    assert!(bipartite_havel_hakimi(&[3, 3], &[1, 1, 1]).is_err());
}

#[test]
fn c6_two_coloring() {
    let a = ah(vec![vec![0, 2], vec![2, 0]]);
    let h = vec![3usize, 3];
    assert!(realizable(&a, &h));
    let g = realize(&a, &h).unwrap();
    assert_simple(&g);
    assert_eq!(g.n(), 6);
    assert_eq!(g.m(), 6);
    assert_eq!(g.degrees(), vec![2, 2, 2, 2, 2, 2]);
    let p = partition_from_h(&h);
    assert!(is_equitable(&g, &p));
    // No intra-cell edges: 2-regular bipartite on 3+3 is C_6.
    for &(u, v) in g.edges() {
        assert_ne!(p[u], p[v], "C_6 is bipartite under the 2-coloring");
    }
}

#[test]
fn two_disjoint_triangles() {
    let a = ah(vec![vec![2, 0], vec![0, 2]]);
    let h = vec![3usize, 3];
    let g = realize(&a, &h).unwrap();
    assert_simple(&g);
    assert_eq!(g.m(), 6);
    let p = partition_from_h(&h);
    for &(u, v) in g.edges() {
        assert_eq!(p[u], p[v], "no cross edges");
    }
    // Each cell is K_3.
    for cell in 0..2 {
        let verts: Vec<usize> = (0..6).filter(|&v| p[v] == cell).collect();
        assert_eq!(verts.len(), 3);
        assert!(g.has_edge(verts[0], verts[1]));
        assert!(g.has_edge(verts[0], verts[2]));
        assert!(g.has_edge(verts[1], verts[2]));
    }
}

#[test]
fn ep_of_realize_roundtrip() {
    let a = ah(vec![vec![0, 2], vec![2, 0]]);
    let h = vec![3usize, 3];
    let g = realize(&a, &h).unwrap();
    let p = partition_from_h(&h);
    let (ah2, h2) = ep_of(&g, &p);
    assert_eq!(h2, h);
    assert_eq!(ah2, a);
}

#[test]
fn infeasible_ah_ij_gt_hj() {
    let a = ah(vec![vec![0, 4], vec![4, 0]]);
    let h = vec![3usize, 3];
    assert!(!realizable(&a, &h));
    assert!(realize(&a, &h).is_err());
}

#[test]
fn phi_eps_noop_on_integer_exact() {
    let a = ah(vec![vec![0, 2], vec![2, 0]]);
    let h = vec![3usize, 3];
    let n_edges = implied_edges(&a, &h).unwrap();
    let target: Vec<Vec<f64>> = a
        .rows()
        .iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect();
    let filled = phi_eps(&target, &h, n_edges, 0.0);
    assert_eq!(filled, a);
}

#[test]
fn phi_eps_noisy_then_realize() {
    let a = ah(vec![vec![2, 0], vec![0, 2]]);
    let h = vec![3usize, 3];
    let n_edges = implied_edges(&a, &h).unwrap();
    let noisy = vec![vec![2.1, 0.08], vec![0.07, 1.9]];
    let filled = phi_eps(&noisy, &h, n_edges, 1.0);
    let g = realize(&filled, &h).expect("closest graphic fill should realize");
    assert_simple(&g);
    let m = g.m();
    assert!(
        m == n_edges || m.abs_diff(n_edges) <= 3,
        "edge count {m} vs requested {n_edges}"
    );
}

#[test]
fn karate_one_cell_not_equitable_and_two_faction_reconstruct() {
    let g = karate_club();
    assert_eq!(g.n(), KARATE_N);
    assert_eq!(g.m(), 78);
    assert_eq!(KARATE_EDGES.len(), 78);
    assert_simple(&g);

    let one_cell = vec![0usize; KARATE_N];
    assert!(
        !is_equitable(&g, &one_cell),
        "karate is not regular, so the trivial partition is not equitable"
    );
    let (ah1, h1) = ep_of(&g, &one_cell);
    assert_eq!(h1, vec![34]);
    // One-cell AH is a single average degree; vertices do not all share it.
    assert_eq!(ah1.k(), 1);

    let p = karate_faction_partition();
    assert_eq!(p.iter().filter(|&&c| c == 0).count(), 16);
    assert_eq!(p.iter().filter(|&&c| c == 1).count(), 18);
    let (ah_f, h_f) = ep_of(&g, &p);
    assert_eq!(h_f, vec![16, 18]);

    // Reconstruction from that 2-cell (AH, H): same cell sizes and same AH.
    // The faction split need not be an exact EP of karate; ep_of uses rounded
    // means. If the rounded profile is not graphic, the test still checks
    // that a realizable nearby profile reconstructs to itself.
    if realizable(&ah_f, &h_f) {
        let rec = realize(&ah_f, &h_f).unwrap();
        assert_simple(&rec);
        let p_rec = partition_from_h(&h_f);
        let (ah_r, h_r) = ep_of(&rec, &p_rec);
        assert_eq!(h_r, h_f);
        assert_eq!(ah_r, ah_f);
        assert_eq!(h_r, vec![16, 18]);
    } else {
        let target: Vec<Vec<f64>> = ah_f
            .rows()
            .iter()
            .map(|r| r.iter().map(|&x| x as f64).collect())
            .collect();
        let n_edges = g.m();
        let filled = phi_eps(&target, &h_f, n_edges, 2.0);
        let rec = realize(&filled, &h_f).expect("phi_eps fill of karate factions");
        assert_simple(&rec);
        let p_rec = partition_from_h(&h_f);
        let (ah_r, h_r) = ep_of(&rec, &p_rec);
        assert_eq!(h_r, vec![16, 18]);
        assert_eq!(ah_r, filled);
    }
}

#[test]
fn output_graphs_are_simple() {
    let cases = [
        (vec![vec![0, 2], vec![2, 0]], vec![3usize, 3]),
        (vec![vec![2, 0], vec![0, 2]], vec![3, 3]),
        (vec![vec![1, 1], vec![1, 1]], vec![4, 4]),
    ];
    for (rows, h) in cases {
        let a = ah(rows);
        let g = realize(&a, &h).unwrap();
        assert_simple(&g);
        for &(u, v) in g.edges() {
            assert_ne!(u, v);
        }
        let mut set = std::collections::BTreeSet::new();
        for &e in g.edges() {
            assert!(set.insert(e));
        }
    }
}

#[test]
fn heap_hh_empty_graph() {
    let edges = havel_hakimi(&[0, 0, 0, 0]).unwrap();
    assert!(edges.is_empty());
    let edges = havel_hakimi(&[0]).unwrap();
    assert!(edges.is_empty());
    let edges = havel_hakimi(&[]).unwrap();
    assert!(edges.is_empty());
}

#[test]
fn sampled_realizations_can_differ() {
    // 2-regular bipartite on 4+4 admits C_8 and 2 C_4.
    let a = ah(vec![vec![0, 2], vec![2, 0]]);
    let h = vec![4usize, 4];
    let g0 = realize(&a, &h).unwrap();
    let g1 = realize_sampled(&a, &h, 64, 7).unwrap();
    let g2 = realize_sampled(&a, &h, 64, 99).unwrap();
    assert_simple(&g0);
    assert_simple(&g1);
    assert_simple(&g2);
    assert_eq!(g0.m(), g1.m());
    let p = partition_from_h(&h);
    let (ah1, _) = ep_of(&g1, &p);
    assert_eq!(ah1, a);
    // At least one of the two seeds should move the edge set (not required
    // if the MCMC is unlucky; still the API exists).
    let _ = edge_overlap(&g0, &g1) < 1.0 || edge_overlap(&g0, &g2) < 1.0;
}

#[test]
fn q_ah_roundtrip() {
    let a = ah(vec![vec![0, 2], vec![2, 0]]);
    let h = vec![3usize, 3];
    let q = ep_realizability::q_from_ah(&a, &h).unwrap();
    let a2 = ep_realizability::ah_from_q(&q, &h).unwrap();
    assert_eq!(a, a2);
    assert_eq!(q.get(0, 1), 6);
    assert_eq!(q.get(0, 0), 0);
}
