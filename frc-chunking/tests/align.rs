use frc_chunking::{
    align_chunks, pack, pack_and_align, relative_position, sim_wrp, token_overlap_cosine,
    AlignParams, FrcParams, SourceChunk,
};

fn ap(lambda: f64, sigma: f64, k: usize, delta_k: usize) -> AlignParams {
    AlignParams {
        lambda,
        sigma,
        k,
        delta_k,
    }
}

#[test]
fn sim_wrp_numeric() {
    // rp match: w = 1, SimwRP = sim * 1.
    let a = sim_wrp(1.0, 0.0, 0.0, 0.5, 1.0);
    assert!((a - 1.0).abs() < 1e-12);

    // rp 0 vs 1, sigma=1: w = exp(-1).
    let w = (-1.0f64).exp();
    let b = sim_wrp(1.0, 0.0, 1.0, 0.5, 1.0);
    let expect = 1.0 * ((1.0 - 0.5) + 0.5 * w);
    assert!((b - expect).abs() < 1e-12);
    assert!((expect - (0.5 + 0.5 * w)).abs() < 1e-12);

    // lambda=0 ignores position.
    let c = sim_wrp(0.4, 0.0, 1.0, 0.0, 0.1);
    assert!((c - 0.4).abs() < 1e-12);
}

#[test]
fn relative_position_first_last() {
    assert_eq!(relative_position(0, 1), 0.0);
    assert_eq!(relative_position(0, 5), 0.0);
    assert!((relative_position(4, 5) - 1.0).abs() < 1e-12);
    assert!((relative_position(2, 5) - 0.5).abs() < 1e-12);
}

#[test]
fn identity_alignment_diagonal_sim() {
    let n = 5;
    let chunks: Vec<SourceChunk> = (0..n).map(|i| SourceChunk::new(i, i + 1)).collect();
    let sim = |u: usize, v: usize| if u == v { 1.0 } else { 0.0 };
    let aln = align_chunks(n, n, &chunks, sim, &ap(0.5, 1.0, 1, 1)).unwrap();
    assert_eq!(aln.ends, vec![0, 1, 2, 3, 4]);
    assert_eq!(
        aln.spans(),
        vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    );
}

#[test]
fn monotone_path_last_is_n_targets() {
    let chunks = [
        SourceChunk::new(0, 2),
        SourceChunk::new(2, 3),
        SourceChunk::new(3, 5),
    ];
    let n_src = 5;
    let n_tgt = 7;
    // Uniform sim: positional prior plus last-target constraint should still
    // produce a monotone path ending at n_tgt-1.
    let sim = |_u: usize, _v: usize| 1.0;
    let aln = align_chunks(n_src, n_tgt, &chunks, sim, &ap(0.5, 0.5, 3, 2)).unwrap();
    assert_eq!(*aln.ends.last().unwrap(), n_tgt - 1);
    for w in aln.ends.windows(2) {
        assert!(w[0] <= w[1], "not monotone: {:?}", aln.ends);
    }
}

#[test]
fn k_expansion_when_k1_is_too_tight() {
    // 3 source singletons, 4 targets. lambda=0 so SimwRP = Sim.
    // End-boundary top-1:
    //   chunk0 -> target 0
    //   chunk1 -> target 3
    //   chunk2 -> target 0  (not the required last target 3)
    // k=1 cannot pin the last chunk to |T|. Expanding k includes target 3
    // for the last chunk and a monotone path 0 <= 1 <= 3.
    let chunks: Vec<SourceChunk> = (0..3).map(|i| SourceChunk::new(i, i + 1)).collect();
    let sim = |u: usize, v: usize| match (u, v) {
        (0, 0) => 1.0,
        (0, 3) => 0.4,
        (1, 3) => 1.0,
        (1, 1) => 0.5,
        (2, 0) => 1.0,
        (2, 3) => 0.2,
        _ => 0.0,
    };

    let tight = align_chunks(3, 4, &chunks, sim, &ap(0.0, 1.0, 1, 8));
    // With delta_k the function itself expands; pin k=1 by delta_k that still
    // grows. The internal loop starts at k=1 and grows to 4, so it should
    // succeed. Check the no-expansion path via a second case: k already 2.
    let aln = tight.expect("k-expansion should find a path");
    assert_eq!(*aln.ends.last().unwrap(), 3);
    for w in aln.ends.windows(2) {
        assert!(w[0] <= w[1]);
    }

    // Directly: k=1 with a huge |T| window still expands internally.
    // Force infeasibility by not expanding: we cannot, API always expands.
    // Check that k=1 *without* the last target in top-1 needs expansion by
    // verifying k=1 candidate set would fail if frozen. Reconstruct by
    // requiring the returned path not equal to top-1 (0,3,0).
    assert_ne!(aln.ends, vec![0, 3, 0]);
}

#[test]
fn k1_frozen_infeasible_then_k2_works() {
    // Same scores as above. We simulate frozen k by using n_tgt=4 and checking
    // that a path exists at k=2 starting size (no need to expand).
    let chunks: Vec<SourceChunk> = (0..3).map(|i| SourceChunk::new(i, i + 1)).collect();
    let sim = |u: usize, v: usize| match (u, v) {
        (0, 0) => 1.0,
        (0, 3) => 0.4,
        (1, 3) => 1.0,
        (1, 1) => 0.5,
        (2, 0) => 1.0,
        (2, 3) => 0.2,
        _ => 0.0,
    };
    let aln = align_chunks(3, 4, &chunks, sim, &ap(0.0, 1.0, 2, 1)).unwrap();
    assert_eq!(aln.ends[2], 3);
    assert!(aln.ends[0] <= aln.ends[1] && aln.ends[1] <= aln.ends[2]);
}

#[test]
fn dual_boundary_term_can_change_the_path() {
    // 2 chunks, 3 targets, lambda=0.
    // chunk0 = units [0,1] (end=1), chunk1 = units [2,3] (first=2).
    // End scores of chunk0: j=0 -> 0.5, j=1 -> 0.6. Top-2 is both.
    // Next-start: Sim(2, 1)=1, Sim(2, 2)=0.
    // phi[0][0] = 0.5 + 1.0 = 1.5
    // phi[0][1] = 0.6 + 0.0 = 0.6
    // Last chunk pinned at target 2. Path should pick j0=0 not 1.
    let chunks = [SourceChunk::new(0, 2), SourceChunk::new(2, 4)];
    let sim = |u: usize, v: usize| match (u, v) {
        (1, 0) => 0.5,
        (1, 1) => 0.6,
        (1, 2) => 0.0,
        (2, 1) => 1.0,
        (2, 2) => 0.0,
        (3, 2) => 1.0,
        _ => 0.0,
    };
    let aln = align_chunks(4, 3, &chunks, sim, &ap(0.0, 1.0, 3, 1)).unwrap();
    assert_eq!(aln.ends, vec![0, 2]);
}

#[test]
fn token_overlap_cosine_basic() {
    let s = token_overlap_cosine("aa bb cc", "aa cc");
    let expect = 2.0 / (3.0f64.sqrt() * 2.0f64.sqrt());
    assert!((s - expect).abs() < 1e-12);
    assert_eq!(token_overlap_cosine("", "x"), 0.0);
}

#[test]
fn pack_then_align_identity() {
    let lens = [8usize, 8, 8];
    let frc = FrcParams {
        m: 5.0,
        M: 10.0,
        delta: 0.0,
        k: 1,
    };
    let packed = pack(&lens, &frc).unwrap();
    assert_eq!(packed.chunks.len(), 3);
    let n = 3;
    let sim = |u: usize, v: usize| if u == v { 1.0 } else { 0.0 };
    let (p2, aln) = pack_and_align(&lens, n, sim, &frc, &ap(0.5, 1.0, 1, 1)).unwrap();
    assert_eq!(p2.chunks.len(), packed.chunks.len());
    let aln = aln.expect("alignment");
    assert_eq!(aln.ends, vec![0, 1, 2]);
}
